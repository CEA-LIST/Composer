import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict
from composer.constants import DEFAULT_TOKENS, IGNORE_INDEX
from composer.data.conversation import conv_templates
from composer.constants import SPECIAL_TOKENS


class CC3M(Dataset):
    """Dataset for simple image-text pairs."""

    def __init__(self, ann_file, img_prefix, tokenizer, img_processor, use_proxy_tokens, conv_temp='llava'):
        super(CC3M, self).__init__()
        self.meta_data = json.load(open(ann_file, "r"))
        self.image_folder = img_prefix
        self.tokenizer = tokenizer
        self.img_processor = img_processor
        self.conv_temp = conv_templates[conv_temp]
        self.seperator_id = self.tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['sep']])[0]
        self.eos_id = self.tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['eos']])[0]
        self.use_proxy_tokens = use_proxy_tokens
        
    def __len__(self):
        return len(self.meta_data)

    def preprocess(self, conversations):
        wrap_reasoning_sequence_fn = lambda reasoning_sequence: f"{SPECIAL_TOKENS['REASONING_START']}{reasoning_sequence}{SPECIAL_TOKENS['REASONING_END']}"
        wrap_answer_sequence_fn = lambda answer_sequence: f"{SPECIAL_TOKENS['ANSWER_START']}{answer_sequence}{SPECIAL_TOKENS['ANSWER_END']}"

        new_conversations = []
        instruct = "Here is an image: {}".format(DEFAULT_TOKENS['image'])
        answer = 'Thank you for the image! How can I assist you with it?'
        new_conversations.append((self.conv_temp.roles[0], instruct))
        new_conversations.append((self.conv_temp.roles[1], answer))
        
        assert len(conversations) % 2 == 0
        for i, conversation in enumerate(conversations):
            chat = conversation['value']
            chat = chat.replace('<image>', '')
            chat = chat.replace('\n', ' ')
            if i % 2 == 1:
                reasoning_sequence = wrap_reasoning_sequence_fn("")
                answer_sequence = wrap_answer_sequence_fn(chat)
                chat = DEFAULT_TOKENS['sep'] + reasoning_sequence + answer_sequence + DEFAULT_TOKENS['sep']
            new_conversations.append((self.conv_temp.roles[i%2], chat))
        prompt = self.conv_temp.get_prompt(new_conversations)

        # tokenize conversations
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True
        ).input_ids[0]
        
        # Mask targets
        targets = input_ids.clone()
        sep_inds = (input_ids == self.seperator_id).nonzero(as_tuple=True)[0]
        assert len(sep_inds) % 2 == 0
        for i in range(0, len(sep_inds), 2):
            pre_sep = 0 if i == 0 else sep_inds[i - 1]
            cur_sep = sep_inds[i]
            targets[pre_sep:cur_sep] = IGNORE_INDEX
        eos_inds = (input_ids == self.eos_id).nonzero(as_tuple=True)[0]
        targets[eos_inds[1:]] = self.eos_id

        # Remove sep token
        mask = input_ids != self.seperator_id
        input_ids = input_ids[mask]
        targets = targets[mask]

        data_dict = dict(
            input_ids=input_ids,
            labels=targets,
            source='cc3m'
        )
        return data_dict

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data_source = self.meta_data[i]
        data_dict = self.preprocess(data_source['conversations'])
        if 'image' in data_source:
            image_file = data_source['image']
            image_folder = self.image_folder
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            image = self.img_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            data_dict['image'] = image
        return data_dict


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoImageProcessor
    from tqdm import tqdm

    tokenizer = AutoTokenizer.from_pretrained(
            "lmsys/vicuna-7b-v1.5",
            cache_dir="cache",
            model_max_length=2048,
            padding_side="right",
            use_fast=False
        )
    num_new_token = tokenizer.add_tokens(list(DEFAULT_TOKENS.values()) + list(SPECIAL_TOKENS.values()), special_tokens=True)
    tokenizer.pad_token = DEFAULT_TOKENS['pad']
    vis_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    dataset = CC3M(
        ann_file="/home/thodemon/workspace/composer-rebuild/dataset/cc3m/chat.json",
        img_prefix="/home/thodemon/workspace/composer-rebuild/dataset/cc3m/images",
        tokenizer=tokenizer,
        img_processor=vis_processor,
        use_proxy_tokens=False,
        conv_temp='llava',
    )

    for i in tqdm(range(len(dataset))):
        out = dataset[i]
        # print(out)
        
        input_ids = out['input_ids']
        target_ids = out['labels']
        # # remove all -100
        target_ids = target_ids[target_ids != -100]
        print(f"input_ids: {tokenizer.decode(input_ids, skip_special_tokens=False)}")
        # print(f"target_ids: {tokenizer.decode(target_ids, skip_special_tokens=False)}")

        # print(f"metadata: {out.metadata}")
        # print("-"*100)

        break
