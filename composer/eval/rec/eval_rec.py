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
from composer.data.datasets.refcoco import RefCOCO, INSTRUCTIONS
from composer.data.datasets.utils import *


class RefCOCOTest(RefCOCO):
    def preprocess(self, data_item):

        convert_bbox_to_region_tokens_fn = lambda bbox: convert_bbox_to_region_tokens(bbox)
        rescale_bbox_fn = lambda bbox: rescale_bbox(bbox, self.resize_size / data_item["width"], self.resize_size / data_item["height"])
        wrap_reasoning_sequence_fn = lambda reasoning_sequence: f"{SPECIAL_TOKENS['REASONING_START']}{reasoning_sequence}{SPECIAL_TOKENS['REASONING_END']}"
        wrap_answer_sequence_fn = lambda answer_sequence: f"{SPECIAL_TOKENS['ANSWER_START']}{answer_sequence}{SPECIAL_TOKENS['ANSWER_END']}"

        def get_bbox_sequence(bbox):
            if bbox is None:
                return f"{SPECIAL_TOKENS['BBOX_START']}{SPECIAL_TOKENS['NO_REGION']}{SPECIAL_TOKENS['BBOX_END']}"
            
            bbox = rescale_bbox_fn(bbox)

            if self.use_proxy_tokens:
                return f"{SPECIAL_TOKENS['BBOX_START']}{convert_bbox_to_region_tokens_fn(bbox)}{SPECIAL_TOKENS['BBOX_END']}"
            else:
                bbox_sequence = f"x={bbox[0]} y={bbox[1]} w={bbox[2]} h={bbox[3]}"
                return f"{SPECIAL_TOKENS['BBOX_START']}{bbox_sequence}{SPECIAL_TOKENS['BBOX_END']}"

        # create target sequence
        def create_object_instance_sequence(object_instance) -> str:
            label = f"{SPECIAL_TOKENS['LABEL_START']}{object_instance['label']}{SPECIAL_TOKENS['LABEL_END']}"
            bbox = object_instance['bbox']
            bbox_sequence = get_bbox_sequence(bbox)
            return f"{SPECIAL_TOKENS['OBJECT_INSTANCE_START']}{label}{bbox_sequence}{SPECIAL_TOKENS['OBJECT_INSTANCE_END']}"
        
        def create_object_recognition_sequence(task) -> str:
            object_instance = task["objectInstance"]
            object_instance_sequence = create_object_instance_sequence(object_instance)
            return f"{SPECIAL_TOKENS['OBJECT_RECOGNITION_START']}{object_instance_sequence}{SPECIAL_TOKENS['OBJECT_RECOGNITION_END']}"
        
        # fake task
        task = {
            "atomicTask": "ObjectRecognition",
            "objectInstance": {
                "label": data_item["caption"],
                "bbox": data_item["gt_bbox"]
            }
        }
        reasoning_sequence = create_object_recognition_sequence(task)

        reasoning_sequence = wrap_reasoning_sequence_fn(reasoning_sequence)
        answer_sequence = wrap_answer_sequence_fn("")

        new_conversations = []
        instruct = "Here is an image: {}".format(DEFAULT_TOKENS['image'])
        answer = 'Thank you for the image! How can I assist you with it?'
        new_conversations.append((self.conv_temp.roles[0], instruct))
        new_conversations.append((self.conv_temp.roles[1], answer))

        question = random.choice(INSTRUCTIONS).format(data_item["caption"])
        new_conversations.append((self.conv_temp.roles[0], question))
        new_conversations.append((self.conv_temp.roles[1], "")) # !!!!!
      
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
            target_sequence=reasoning_sequence + answer_sequence
        )

        return data_dict

    def __getitem__(self, i):
        img_id, ann = self.meta_data[i]
        img_info = self.coco.loadImgs(img_id)[0]
        gt_bbox = ann["bbox"] # [x, y, w, h]

        data_item = {
            "caption": img_info["caption"], 
            "width": img_info["width"],
            "height": img_info["height"],
            "gt_bbox": gt_bbox
        }

        data_dict = self.preprocess(data_item)

        data_dict["gt_bbox"] = gt_bbox
        data_dict["width"] = img_info["width"]
        data_dict["height"] = img_info["height"]

        image_file = img_info["file_name"]
        image_folder = self.coco_dataset.root
        image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
        image = self.img_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        data_dict['image'] = image
        data_dict['image_file'] = image_file
        data_dict['id'] = ann['id']

        return data_dict


def custom_collate_fn(batch):
    assert len(batch) == 1
    input_ids = batch[0]['input_ids']
    target_sequence = batch[0]['target_sequence']
    gt_bbox = batch[0]['gt_bbox']
    image = batch[0]['image'].unsqueeze(dim=0)
    image_file = batch[0]['image_file']
    id = batch[0]['id']
    width = batch[0]["width"]
    height = batch[0]["height"]
    return input_ids, image, target_sequence, gt_bbox, image_file, id, width, height


def eval_model(args):
    print(args)

    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = ComposerModel.from_pretrained(model_name).cuda()
    
    model.init_special_token_id(tokenizer)
    vis_processor = CLIPImageProcessor.from_pretrained(model_name)

    dataset = RefCOCOTest(
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

    for input_ids, image, target_sequence, gt_bbox, image_file, id, width, height in tqdm(dataloader):
        input_ids = input_ids.cuda()
        image = image.cuda()

        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image,
                use_cache=True,
                do_sample=False,
                max_new_tokens=256,
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
            "id": id,
            "prediction": outputs,
            "target_sequence": target_sequence,
            "image_file": image_file,
            "gt_bbox": gt_bbox,
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
    parser.add_argument("--ann_file", type=str, default="dataset/refcoco/val/refcoco_val.json")
    parser.add_argument("--img_prefix", type=str, default="dataset/coco/train2017")
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