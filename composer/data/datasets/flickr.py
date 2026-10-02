import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection


from composer.constants import DEFAULT_TOKENS, IGNORE_INDEX
from composer.data.conversation import conv_templates
from composer.constants import *
from composer.data.datasets.utils import *


INSTRUCTIONS = [
    "Give me a concise description of the image.",
    "Please briefly summarize the content of this image.",
    "What does this picture show? Please summarize briefly.",
    "Can you give me a quick overview of what's depicted in this image?",
    "Could you describe the key elements in this photograph?",
    "Offer a brief explanation of what this image represents.",
    "Sum up the contents of this picture in one or two sentences."
]

class Flickr30k(Dataset):

    def __init__(self, ann_file, img_prefix, tokenizer, img_processor, use_proxy_tokens, conv_temp='llava') -> None:
        super(Flickr30k, self).__init__()

        self.coco_dataset = CocoDetection(img_prefix, ann_file)
        self.coco = self.coco_dataset.coco
        self.meta_data = self.coco.getImgIds()

        self.image_folder = img_prefix
        self.tokenizer = tokenizer
        self.img_processor = img_processor
        self.conv_temp = conv_templates[conv_temp]
        self.seperator_id = self.tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['sep']])[0]
        self.eos_id = self.tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['eos']])[0]

        self.use_proxy_tokens = use_proxy_tokens
        
    def __len__(self):
        return len(self.meta_data)
    
    def preprocess(self, data_item):

        wrap_reasoning_sequence_fn = lambda reasoning_sequence: f"{SPECIAL_TOKENS['REASONING_START']}{reasoning_sequence}{SPECIAL_TOKENS['REASONING_END']}"
        wrap_answer_sequence_fn = lambda answer_sequence: f"{SPECIAL_TOKENS['ANSWER_START']}{answer_sequence}{SPECIAL_TOKENS['ANSWER_END']}"

        new_conversations = []
        instruct = "Here is an image: {}".format(DEFAULT_TOKENS['image'])
        answer = 'Thank you for the image! How can I assist you with it?'
        new_conversations.append((self.conv_temp.roles[0], instruct))
        new_conversations.append((self.conv_temp.roles[1], answer))

        question = random.choice(INSTRUCTIONS).format(data_item["caption"])
        reasoning_sequence = wrap_reasoning_sequence_fn("")
        answer_sequence = wrap_answer_sequence_fn(data_item["caption"])
        answer = DEFAULT_TOKENS["sep"] + reasoning_sequence + answer_sequence + DEFAULT_TOKENS["sep"]
        new_conversations.append((self.conv_temp.roles[0], question))
        new_conversations.append((self.conv_temp.roles[1], answer))
      
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
            source='flickr'
        )
        return data_dict
    
    def __getitem__(self, i):
        
        img_id = self.meta_data[i]
        img_info = self.coco.loadImgs(img_id)[0]

        data_item = {
            "caption": img_info["caption"],
        }

        data_dict = self.preprocess(data_item)
        
        image_file = img_info["file_name"]
        image_folder = self.coco_dataset.root
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
    num_new_token = tokenizer.add_tokens(list(DEFAULT_TOKENS.values()) + PROXY_TOKENS + list(SPECIAL_TOKENS.values()), special_tokens=True)
    tokenizer.pad_token = DEFAULT_TOKENS['pad']
    vis_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    dataset = Flickr30k(
        ann_file="/home/thodemon/workspace/composer-rebuild/dataset/flickr30k/ann/flickr30k_entities_train.json",
        img_prefix="/home/thodemon/workspace/composer-rebuild/dataset/flickr30k/images",
        tokenizer=tokenizer,
        img_processor=vis_processor,
        conv_temp='default',
        use_proxy_tokens=True
    )

    for i in tqdm(range(len(dataset))):
        out = dataset[i]
        # print(out)
        
        input_ids = out['input_ids']
        target_ids = out['labels']
        
        print(f"input_ids: {input_ids}")
        print(f"target_ids: {target_ids}")
        # # remove all -100
        target_ids = target_ids[target_ids != -100]
        print(f"input_ids: {tokenizer.decode(input_ids, skip_special_tokens=False)}")
        print(f"target_ids: {tokenizer.decode(target_ids, skip_special_tokens=False)}")

        # print(f"metadata: {out.metadata}")
        # print("-"*100)

        break
