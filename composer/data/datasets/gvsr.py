import os
import json
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection

from composer.constants import DEFAULT_TOKENS, IGNORE_INDEX, SPECIAL_TOKENS
from composer.data.conversation import conv_templates
from composer.data.datasets.utils import *


INSTRUCTIONS = [
    "Is the following statement true or false? {}",
    "Determine whether this statement is true or false: {}",
    "Assess the truthfulness of this statement: {}",
    "Classify this as true or false: {}",
    "Does the statement below hold true or not? {}",
]

class GVSR(Dataset):

    def __init__(self, ann_file, img_prefix, tokenizer, img_processor, use_proxy_tokens, conv_temp='llava') -> None:
        super(GVSR, self).__init__()

        with open(ann_file, "r") as f:
            self.meta_data = [json.loads(line) for line in f]
        self.image_folder = img_prefix

        self.tokenizer = tokenizer
        self.img_processor = img_processor
        self.conv_temp = conv_templates[conv_temp]
        self.seperator_id = self.tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['sep']])[0]
        self.eos_id = self.tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['eos']])[0]

       
        self.resize_size = 224

        self.use_proxy_tokens = use_proxy_tokens

    def __len__(self):
        return len(self.meta_data)
    
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
        
        def create_subject_instance_sequence(subject_instance) -> str:
            label = f"{SPECIAL_TOKENS['LABEL_START']}{subject_instance['label']}{SPECIAL_TOKENS['LABEL_END']}"
            bbox = subject_instance['bbox']
            bbox_sequence = get_bbox_sequence(bbox)
            return f"{SPECIAL_TOKENS['SUBJECT_INSTANCE_START']}{label}{bbox_sequence}{SPECIAL_TOKENS['SUBJECT_INSTANCE_END']}"

        def create_relation_instance_sequence(relation_instance) -> str:
            relation = relation_instance["relation"]
            relation_sequence = f"{SPECIAL_TOKENS['RELATION_START']}{relation}{SPECIAL_TOKENS['RELATION_END']}"
            return f"{SPECIAL_TOKENS['RELATION_INSTANCE_START']}{relation_sequence}{SPECIAL_TOKENS['RELATION_INSTANCE_END']}"
        
        def create_object_recognition_sequence(task) -> str:
            object_instance = task["objectInstance"]
            object_instance_sequence = create_object_instance_sequence(object_instance)
            return f"{SPECIAL_TOKENS['OBJECT_RECOGNITION_START']}{object_instance_sequence}{SPECIAL_TOKENS['OBJECT_RECOGNITION_END']}"
        
        def create_spatial_relationship_sequence(task) -> str:
            object_instance = task["objectInstance"]
            subject_instance = task["subjectInstance"]
            object_instance_sequence = create_object_instance_sequence(object_instance)
            subject_instance_sequence = create_subject_instance_sequence(subject_instance)
            relation_instance = task["relationInstance"]
            relation_sequence = create_relation_instance_sequence(relation_instance)
            return f"{SPECIAL_TOKENS['SPATIAL_RELATIONSHIP_START']}{object_instance_sequence}{relation_sequence}{subject_instance_sequence}{SPECIAL_TOKENS['SPATIAL_RELATIONSHIP_END']}"

        atomic_tasks = [
            {
                "atomicTask": "ObjectRecognition",
                "objectInstance": data_item["objectInstance"]
            },
            {
                "atomicTask": "ObjectRecognition",
                "objectInstance": data_item["subjectInstance"]
            },
            {
                "atomicTask": "SpatialRelationship",
                "objectInstance": data_item["objectInstance"],
                "subjectInstance": data_item["subjectInstance"],
                "relationInstance": data_item["relationInstance"]
            }
        ]  

        reasoning_sequence = ""
        for atomic_task in atomic_tasks:
            if atomic_task["atomicTask"] == "ObjectRecognition":
                reasoning_sequence += create_object_recognition_sequence(atomic_task)
            elif atomic_task["atomicTask"] == "SpatialRelationship":
                reasoning_sequence += create_spatial_relationship_sequence(atomic_task)
            else:
                raise ValueError(f"Unknown atomic task type: {atomic_task['atomicTask']}")

        reasoning_sequence = wrap_reasoning_sequence_fn(reasoning_sequence)
        answer = "Yes." if data_item["label"] == 1 else "No."
        answer_sequence = wrap_answer_sequence_fn(answer)

        new_conversations = []
        instruct = "Here is an image: {}".format(DEFAULT_TOKENS['image'])
        answer = 'Thank you for the image! How can I assist you with it?'
        new_conversations.append((self.conv_temp.roles[0], instruct))
        new_conversations.append((self.conv_temp.roles[1], answer))

        question = random.choice(INSTRUCTIONS).format(data_item["caption"])
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
            source='gvsr'
        )
        return data_dict
    

    def __getitem__(self, i):

        ann = self.meta_data[i]

        data_item = {
            "caption": ann["caption"], 
            "width": ann["width"],
            "height": ann["height"],
            "objectInstance": {
                "label": ann["ref_exp"]["labels"][0],
                "bbox": ann["ref_exp"]["bboxes"][0]
            },
            "subjectInstance": {
                "label": ann["ref_exp"]["labels"][1],
                "bbox": ann["ref_exp"]["bboxes"][1]
            },
            "relationInstance": {
                "relation": ann["relation"]
            },
            "label": ann["label"] 
        }

        data_dict = self.preprocess(data_item)

        image_file = ann["image_file"]
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
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
            use_fast=False,
        )
    num_new_token = tokenizer.add_tokens(list(DEFAULT_TOKENS.values()) + PROXY_TOKENS + list(SPECIAL_TOKENS.values()), special_tokens=True)
    tokenizer.pad_token = DEFAULT_TOKENS['pad']
    vis_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    dataset = GVSR(
        ann_file="/home/thodemon/workspace/composer-rebuild/dataset/gvsr/ann/gvsr.jsonl",
        img_prefix="/home/thodemon/workspace/datasets/coco/train2017",
        tokenizer=tokenizer,
        img_processor=vis_processor,
        use_proxy_tokens=False,
        conv_temp='default'
    )

    for i in tqdm(range(len(dataset))):
        out = dataset[i]
        print(out)
        
        input_ids = out['input_ids']
        labels = out['labels']

        print(f"input_ids:\t {input_ids}")
        print(f"labels:\t {labels}")

        # # remove all -100
        labels = labels[labels != -100]
        print(f"input_ids:\t {tokenizer.decode(input_ids, skip_special_tokens=False)}")
        print(f"labels:\t {tokenizer.decode(labels, skip_special_tokens=False)}")

        # # print(f"metadata: {out.metadata}")
        # # print("-"*100)
        break