import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection

from composer.constants import DEFAULT_TOKENS, IGNORE_INDEX, SPECIAL_TOKENS
from composer.data.conversation import conv_templates
from composer.data.datasets.utils import *


INSTRUCTIONS = [
    "Locate {} in the image.",
    "Can you spot {} in the photograph?",
    "Identify where {} is located in the picture.",
    "Please detect {} in the picture.",
    "Which region matches the description {}?",
    "Please identify the object that corresponds to {}."
]


class RefCOCO(Dataset):

    def __init__(self, ann_file, img_prefix, tokenizer, img_processor, use_proxy_tokens, conv_temp='llava') -> None:
        super(RefCOCO, self).__init__()

        self.coco_dataset = CocoDetection(img_prefix, ann_file)
        self.coco = self.coco_dataset.coco

        self.meta_data = []

        for img_id in self.coco.getImgIds():
            ann = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            assert len(ann) == 1, "Each image should have exactly one annotation"
            self.meta_data.append((img_id, ann[0]))

        self.image_folder = img_prefix
        self.tokenizer = tokenizer
        self.img_processor = img_processor
        self.conv_temp = conv_templates[conv_temp]
        self.seperator_id = self.tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['sep']])[0]
        self.eos_id = self.tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['eos']])[0]

        self.resize_size = 224

        self.use_proxy_tokens = use_proxy_tokens

        self._avg_iou_reconstru = 0
        self._invalid = 0

    def get_avg_iou_reconstru(self, l):
        l = l - self._invalid
        print("invalid: ", self._invalid)
        return (self._avg_iou_reconstru / l) * 100
    
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
        
        def create_object_recognition_sequence(task) -> str:
            object_instance = task["objectInstance"]
            object_instance_sequence = create_object_instance_sequence(object_instance)
            return f"{SPECIAL_TOKENS['OBJECT_RECOGNITION_START']}{object_instance_sequence}{SPECIAL_TOKENS['OBJECT_RECOGNITION_END']}"
        
        # fake task
        task = {
            "atomicTask": "ObjectRecognition",
            "objectInstance": {
                "label": data_item["caption"],
                "bbox": data_item["bbox"]
            }
        }

    #    ###### DEBUG ######
    #     _rescaled_bbox = rescale_bbox_fn(data_item["bbox"])
    #     _patch_indices, _patch_info_dict = convert_bbox_to_patch_indices(_rescaled_bbox, 16, 16, (224, 224))
    #     _reconstructed_bbox = convert_patch_indices_to_bbox_xywh(_patch_info_dict)
    #     # print(_rescaled_bbox)
    #     # print(_patch_indices)
    #     # print(_reconstructed_bbox)
    #     # exit()

    #     import torch
    #     from torchvision.ops import box_iou, box_convert

    #     iou_reconstru = box_iou(
    #         box_convert(torch.tensor(_rescaled_bbox).unsqueeze(0) ,"xywh", "xyxy"),
    #         box_convert(torch.tensor(_reconstructed_bbox).unsqueeze(0) ,"xywh", "xyxy")
    #     )[0][0]

    #     if torch.isnan(iou_reconstru).any():
    #         self._invalid += 1
    #     else: 
    #         self._avg_iou_reconstru += iou_reconstru.item()
    #         print("reconstru iou: ", iou_reconstru)

    #     print("label: ", data_item["caption"])
    #     print("gt_bbox:", data_item["bbox"]) 
    #     print("rescaled_gt_bbox:", _rescaled_bbox)
    #     print(_patch_indices)
    #     print("rescaled RECONSTRU bbox", _reconstructed_bbox)
        
    #     ###### DEBUG ######

        reasoning_sequence = create_object_recognition_sequence(task)

        reasoning_sequence = wrap_reasoning_sequence_fn(reasoning_sequence)
        answer_sequence = wrap_answer_sequence_fn("")

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
            source='refcoco'
        )
        return data_dict
    

    def __getitem__(self, i):
        img_id, ann = self.meta_data[i]
        img_info = self.coco.loadImgs(img_id)[0]
        
        data_item = {
            "bbox": ann["bbox"],  # [x, y, w, h],
            "caption": img_info["caption"], 
            "width": img_info["width"],
            "height": img_info["height"]
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
            use_fast=False,
        )

    num_new_token = tokenizer.add_tokens(list(DEFAULT_TOKENS.values()) + PROXY_TOKENS + list(SPECIAL_TOKENS.values()), special_tokens=True)
    tokenizer.pad_token = DEFAULT_TOKENS['pad']
    vis_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    dataset = RefCOCO(
        ann_file="/home/thodemon/workspace/composer-rebuild/dataset/refcoco/ann/refcoco_testA.json",
        img_prefix="/home/thodemon/workspace/datasets/coco/train2017",
        tokenizer=tokenizer,
        img_processor=vis_processor,
        use_proxy_tokens=True,
        conv_temp='llava'
    )
    
    # dataset[678]
    print(len(dataset))
    # N = 5000
    # for i in tqdm(range(N)):
    #     out = dataset[i]
        
    #     # print(out)
        
    #     # input_ids = out['input_ids']
    #     # labels = out['labels']

    #     # print(f"input_ids:\t {input_ids}")
    #     # print(f"labels:\t {labels}")

    #     # # # remove all -100
    #     # labels = labels[labels != -100]
    #     # print(f"input_ids:\t {tokenizer.decode(input_ids, skip_special_tokens=False)}")
    #     # print(f"labels:\t {tokenizer.decode(labels, skip_special_tokens=False)}")

    #     # print(f"metadata: {out.metadata}")
    #     # print("-"*100)
        

    # print(dataset.get_avg_iou_reconstru(N))