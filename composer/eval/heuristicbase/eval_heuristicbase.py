
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
        
        atomic_tasks = data_item["atomic_tasks"]

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
        
        def create_subject_instance_sequence(subject_instance) -> str:
            label = f"{SPECIAL_TOKENS['LABEL_START']}{subject_instance['label']}{SPECIAL_TOKENS['LABEL_END']}"
            bbox = subject_instance['bbox']
            bbox_sequence = get_bbox_sequence(bbox)
            return f"{SPECIAL_TOKENS['SUBJECT_INSTANCE_START']}{label}{bbox_sequence}{SPECIAL_TOKENS['SUBJECT_INSTANCE_END']}"

        def create_relation_instance_sequence(relation_instance) -> str:
            relation = relation_instance["relation"]
            relation_sequence = f"{SPECIAL_TOKENS['RELATION_START']}{relation}{SPECIAL_TOKENS['RELATION_END']}"
            return f"{SPECIAL_TOKENS['RELATION_INSTANCE_START']}{relation_sequence}{SPECIAL_TOKENS['RELATION_INSTANCE_END']}"

        def create_attribute_color_instance_sequence(attribute_color_instance) -> str:
            color = attribute_color_instance["color"]
            color_sequence = f"{SPECIAL_TOKENS['COLOR_START']}{color}{SPECIAL_TOKENS['COLOR_END']}"
            return f"{SPECIAL_TOKENS['ATTRIBUTE_COLOR_INSTANCE_START']}{color_sequence}{SPECIAL_TOKENS['ATTRIBUTE_COLOR_INSTANCE_END']}"

        def create_spatial_position_instance_sequence(spatial_position_instance) -> str:
            spatial_position = spatial_position_instance["spatialPosition"]
            spatial_position_sequence = f"{SPECIAL_TOKENS['SPATIAL_POSITION_START']}{spatial_position}{SPECIAL_TOKENS['SPATIAL_POSITION_END']}"
            bbox = spatial_position_instance["bbox"]
            bbox_sequence = get_bbox_sequence(bbox)
            return f"{SPECIAL_TOKENS['SPATIAL_POSITION_INSTANCE_START']}{spatial_position_sequence}{bbox_sequence}{SPECIAL_TOKENS['SPATIAL_POSITION_INSTANCE_END']}"

        def create_object_recognition_sequence(task) -> str:
            object_instance = task["objectInstance"]
            object_instance_sequence = create_object_instance_sequence(object_instance)
            return f"{SPECIAL_TOKENS['OBJECT_RECOGNITION_START']}{object_instance_sequence}{SPECIAL_TOKENS['OBJECT_RECOGNITION_END']}"
        
        def create_attribute_color_sequence(task) -> str:
            object_instance = task["objectInstance"]
            object_instance_sequence = create_object_instance_sequence(object_instance)
            attribute_color_instance = task["attributeColorInstance"]
            attribute_color_instance_sequence = create_attribute_color_instance_sequence(attribute_color_instance)
            return f"{SPECIAL_TOKENS['ATTRIBUTE_COLOR_START']}{object_instance_sequence}{attribute_color_instance_sequence}{SPECIAL_TOKENS['ATTRIBUTE_COLOR_END']}"
        
        def create_spatial_relationship_sequence(task) -> str:
            object_instance = task["objectInstance"]
            subject_instance = task["subjectInstance"]
            object_instance_sequence = create_object_instance_sequence(object_instance)
            subject_instance_sequence = create_subject_instance_sequence(subject_instance)
            relation_instance = task["relationInstance"]
            relation_sequence = create_relation_instance_sequence(relation_instance)
            return f"{SPECIAL_TOKENS['SPATIAL_RELATIONSHIP_START']}{object_instance_sequence}{relation_sequence}{subject_instance_sequence}{SPECIAL_TOKENS['SPATIAL_RELATIONSHIP_END']}"

        def create_spatial_position_sequence(task) -> str:
            object_instance = task["objectInstance"]
            object_instance_sequence = create_object_instance_sequence(object_instance)
            spatial_position_instance = task["spatialPositionInstance"]
            spatial_position_instance_sequence = create_spatial_position_instance_sequence(spatial_position_instance)
            return f"{SPECIAL_TOKENS['SPATIAL_POSITION_START']}{spatial_position_instance_sequence}{object_instance_sequence}{SPECIAL_TOKENS['SPATIAL_POSITION_END']}"

        reasoning_sequence = ""   

        for atomic_task in atomic_tasks:
            if atomic_task["atomicTask"] == "ObjectRecognition":
                reasoning_sequence += create_object_recognition_sequence(atomic_task)
            elif atomic_task["atomicTask"] == "AttributeColor":
                reasoning_sequence += create_attribute_color_sequence(atomic_task)
            elif atomic_task["atomicTask"] == "SpatialRelationship":
                reasoning_sequence += create_spatial_relationship_sequence(atomic_task)
            elif atomic_task["atomicTask"] == "SpatialPosition":
                reasoning_sequence += create_spatial_position_sequence(atomic_task)
            else:
                raise ValueError(f"Unknown atomic task type: {atomic_task['atomicTask']}")

        reasoning_sequence = wrap_reasoning_sequence_fn(reasoning_sequence)
        answer_sequence = wrap_answer_sequence_fn(data_item["answer"])

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
            target_sequence=reasoning_sequence + answer_sequence
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
    parser.add_argument("--ann_file", type=str, default="dataset/heuristicbase/val/ann.jsonl")
    parser.add_argument("--img_prefix", type=str, default="dataset/heuristicbase/val/images")
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