import copy
import torch
import numpy as np
from torch.utils.data import ConcatDataset

from composer.utils import load_config_from_yaml
from composer.data.datasets.cc3m import CC3M
from composer.data.datasets.heuristicbase import Heuristicbase
from composer.data.datasets.flickr import Flickr30k
from composer.data.datasets.refcoco import RefCOCO
from composer.data.datasets.coco import COCO
from composer.data.datasets.vg import SingleRoundVG
from composer.data.datasets.gvsr import GVSR

def build_multi_datasets(dataset_cfg_file, use_proxy_tokens, tokenizer=None, **kwargs):
    dataset_cfgs = load_config_from_yaml(dataset_cfg_file)
    dataset_cfgs = dataset_cfgs['datasets']
    assert isinstance(dataset_cfgs, list)
    datasets = [build_dataset(cfg, use_proxy_tokens, tokenizer=tokenizer, **kwargs) for cfg in dataset_cfgs]
    return ConcatDataset(datasets)


def build_dataset(dataset_cfg, use_proxy_tokens, tokenizer=None, **kwargs):
    dataset_type = dataset_cfg.pop('type')
    ratio = dataset_cfg.pop('ratio', 1)
    conv_temp = dataset_cfg.pop('conv_temp', 'llava')

    if dataset_type == 'cc3m':
        dataset = CC3M(**dataset_cfg, tokenizer=tokenizer, img_processor=kwargs['img_processor'], use_proxy_tokens=use_proxy_tokens, conv_temp=conv_temp)
    elif dataset_type == 'flickr':
        dataset = Flickr30k(**dataset_cfg, tokenizer=tokenizer, img_processor=kwargs['img_processor'], use_proxy_tokens=use_proxy_tokens, conv_temp=conv_temp)
    elif dataset_type == 'heuristicbase':
        dataset = Heuristicbase(**dataset_cfg, tokenizer=tokenizer, img_processor=kwargs['img_processor'], use_proxy_tokens=use_proxy_tokens, conv_temp=conv_temp)
    elif dataset_type == 'refcoco':
        dataset = RefCOCO(**dataset_cfg, tokenizer=tokenizer, img_processor=kwargs['img_processor'], use_proxy_tokens=use_proxy_tokens, conv_temp=conv_temp)
    elif dataset_type == 'coco':
        dataset = COCO(**dataset_cfg, tokenizer=tokenizer, img_processor=kwargs['img_processor'], use_proxy_tokens=use_proxy_tokens, conv_temp=conv_temp)
    elif dataset_type == 'vg':
        dataset = SingleRoundVG(**dataset_cfg, tokenizer=tokenizer, img_processor=kwargs['img_processor'], use_proxy_tokens=use_proxy_tokens, conv_temp=conv_temp)
    elif dataset_type == 'gvsr':
        dataset = GVSR(**dataset_cfg, tokenizer=tokenizer, img_processor=kwargs['img_processor'], use_proxy_tokens=use_proxy_tokens, conv_temp=conv_temp)
    else:
        raise NotImplementedError(f"Dataset type {dataset_type} not implemented")

    if ratio < 1:
        print(f'randomly sample {ratio} of the dataset {dataset_type}: {int(ratio * len(dataset))}')
        random_indices = np.random.choice(len(dataset), int(ratio * len(dataset)), replace=False)
        subsample_dataset = torch.utils.data.Subset(dataset, random_indices)
        return subsample_dataset

    return dataset


if __name__ == '__main__':
    # for quick test
    from transformers import AutoTokenizer, CLIPImageProcessor
    from composer.constants import DEFAULT_TOKENS, PROXY_TOKENS, SPECIAL_TOKENS

    tokenizer = AutoTokenizer.from_pretrained(
            "lmsys/vicuna-7b-v1.5",
            cache_dir="cache",
            model_max_length=2048,
            padding_side="right",
            use_fast=False
        )
    num_new_token = tokenizer.add_tokens(list(DEFAULT_TOKENS.values()) + PROXY_TOKENS + list(SPECIAL_TOKENS.values()), special_tokens=True)

    tokenizer.pad_token = DEFAULT_TOKENS['pad']
    vis_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

    dataset_cfg_file = 'composer/data/configs/vl_pretrain_debug.yaml'
    train_datasets = build_multi_datasets(dataset_cfg_file, use_proxy_tokens=True, tokenizer=tokenizer, img_processor=vis_processor)
    print(len(train_datasets))
    # train_datasets[0]
    import random
    for i in range(10):
        ind = random.randint(0, len(train_datasets))
        train_datasets[ind]

