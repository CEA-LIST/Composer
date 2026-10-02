
import os
import torch
import json
import random
import argparse
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, CLIPImageProcessor
from PIL import Image
from tqdm import tqdm

from composer.utils import init_distributed_mode
from composer.constants import *
from composer.model.composer import ComposerModel
from composer.data.datasets.heuristicbase import Heuristicbase
from composer.data.datasets.utils import *


class HeuristicbaseTest(Heuristicbase):
    def preprocess(self, data_item):
 
        new_conversations = []
        instruct = "Here is an image: {}".format(DEFAULT_TOKENS['image'])
        answer = 'Thank you for the image! How can I assist you with it?'
        new_conversations.append((self.conv_temp.roles[0], instruct))
        new_conversations.append((self.conv_temp.roles[1], answer))

        question = DEFAULT_TOKENS["ground_and_reason"] + data_item["question"]
        new_conversations.append((self.conv_temp.roles[0], question))
        new_conversations.append((self.conv_temp.roles[1], "")) # empty string for answer
      
        prompt = self.conv_temp.get_prompt(new_conversations)

        # tokenize conversations
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True
        ).input_ids
        
    
        data_dict = dict(
            input_ids=input_ids,
            target_sequence=data_item["answer"]
        )

        return data_dict

    def __getitem__(self, i):
        data_item = self.meta_data[i]
        data_dict = self.preprocess(data_item)

        image_file = f"{data_item['image_id']}.jpg"
        image_folder = self.image_folder
        image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')

        data_dict["width"], data_dict["height"] = image.size

        image = self.img_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        data_dict['image'] = image
        data_dict['question_id'] = data_item['question_id']
        data_dict['image_file'] = image_file
        return data_dict


def custom_collate_fn(batch):
    assert len(batch) == 1
    input_ids = batch[0]['input_ids']
    image = batch[0]['image'].unsqueeze(dim=0)
    target_sequence = batch[0]['target_sequence']
    question_id = batch[0]['question_id']
    image_file = batch[0]['image_file']
    width = batch[0]["width"]
    height = batch[0]["height"]
    return input_ids, image, target_sequence, question_id, image_file, width, height


def eval_model(args):
    print(args)

    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = ComposerModel.from_pretrained(model_name).cuda()
    
    model.init_special_token_id(tokenizer)
    vis_processor = CLIPImageProcessor.from_pretrained(model_name)

    dataset = HeuristicbaseTest(
        ann_file=args.ann_file,
        img_prefix=args.img_prefix,
        tokenizer=tokenizer,
        img_processor=vis_processor,
        use_proxy_tokens=True,
        conv_temp='llava'
    )

    distributed_sampler = DistributedSampler(dataset, rank=args.rank, shuffle=False)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size_per_gpu, num_workers=4,
        sampler=distributed_sampler, collate_fn=custom_collate_fn)

    results = []
    count = 0
    flush_interval = 100
    results_path = f"{args.result_dir}/{os.path.basename(args.model_name)}_{args.rank}.jsonl"

    for input_ids, image, target_sequence, question_id, image_file, width, height in tqdm(dataloader):
        input_ids = input_ids.cuda()
        image = image.cuda()

        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image,
                use_cache=True,
                do_sample=False,
                max_new_tokens=2048,
                return_dict_in_generate=True,
                output_hidden_states=True,
                generation_config=model.generation_config
            )

        output_ids = outputs.sequences
        input_token_len = input_ids.shape[1]
        output_ids = output_ids[:, input_token_len:]

        if args.decode:
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        else: 
            outputs = output_ids.tolist()[0]

        result = {
            "prediction": outputs,
            "question_id": question_id,
            "target_sequence": target_sequence,
            "image_file": image_file,
            "width": width,
            "height": height
        }
        results.append(result)

        count += 1
        if args.limit_per_gpu is not None:
            if count >= args.limit_per_gpu:
                break

        if count % flush_interval == 0:
            os.makedirs(args.result_dir, exist_ok=True)
            
            with open(results_path, 'a') as json_file:
                for result in results:
                    json.dump(result, json_file)
                    json_file.write('\n')

            results = []

    if len(results) > 0:
        os.makedirs(args.result_dir, exist_ok=True)
        with open(results_path, 'a') as json_file:
            for result in results:
                json.dump(result, json_file)
                json_file.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--ann_file", type=str, default="dataset/heuristicbase/test/ann.jsonl")
    parser.add_argument("--img_prefix", type=str, default="dataset/heuristicbase/test/images")
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--limit_per_gpu", type=int, default=None)
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--batch_size_per_gpu", type=int, default=1)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()
    init_distributed_mode(args)

    eval_model(args)