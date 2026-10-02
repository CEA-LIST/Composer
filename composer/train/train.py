# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
import pathlib
import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
from transformers import LlamaConfig, CLIPImageProcessor, CLIPVisionConfig

from composer.model.composer import ComposerConfig, ComposerModel
from composer.data.build import build_multi_datasets
from composer.data.collator import DataCollatorForHybridDataset
from composer.constants import DEFAULT_TOKENS, PROXY_TOKENS, SPECIAL_TOKENS
from composer.train.composer_trainer import ComposerTrainer


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    llm: Optional[str] = field(default=None)
    vision_encoder: Optional[str] = field(default=None)
   

@dataclass
class DataArguments:
    dataset_config: str = field(default='composer/data/configs/vl_pretrain.yaml')


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    freeze_llm: bool = field(default=False)
    freeze_vision_encoder: bool = field(default=True)
    freeze_vl_bridge: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    ddp_find_unused_parameters: bool = field(default=True)
    model_max_length: int = field(default=512)
    use_custom_lr: bool = field(default=False)
    custom_lr_params: tuple = field(default=('llm'))
    custom_lr: float = field(default=2e-5)
    group_by_data_source: Optional[bool] = field(default=True)
    report_to: str = field(default="none")
    use_proxy_tokens: bool = field(default=True)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(f"Training arguments: {training_args}")
        
    if model_args.model_name_or_path:
        # initialize from pretrained composer model
        # to check loader tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        model = ComposerModel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
        vis_processor = CLIPImageProcessor.from_pretrained(model_args.model_name_or_path)
        model.init_special_token_id(tokenizer)
    elif model_args.llm and model_args.vision_encoder:
        # initialize from scratch
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.llm,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        if training_args.use_proxy_tokens:
            print("Adding proxy tokens to tokenizer")
            num_new_token = tokenizer.add_tokens(list(DEFAULT_TOKENS.values()) + PROXY_TOKENS + list(SPECIAL_TOKENS.values()), special_tokens=True)
        else:
            num_new_token = tokenizer.add_tokens(list(DEFAULT_TOKENS.values()) + list(SPECIAL_TOKENS.values()), special_tokens=True)
        tokenizer.pad_token = DEFAULT_TOKENS['pad']
        vis_processor = CLIPImageProcessor.from_pretrained(model_args.vision_encoder)
        model_cfg = ComposerConfig(
            llm_cfg=LlamaConfig.from_pretrained(model_args.llm),
            vision_encoder_cfg=CLIPVisionConfig.from_pretrained(model_args.vision_encoder),
            num_new_token=num_new_token,
            use_proxy_tokens=training_args.use_proxy_tokens
        )
        model = ComposerModel(
            model_cfg,
            pretrained_vision_encoder=model_args.vision_encoder,
            pretrained_llm=model_args.llm,
        )
        model.init_special_token_id(tokenizer)
        model.llm.config.use_cache = False
        model.generation_config = model.llm.generation_config
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.do_sample = True
    else:
        raise ValueError("Should specify either the pretrained model or the model config.")

    if training_args.freeze_vision_encoder:
        model.freeze_vision_encoder()
    if training_args.freeze_vl_bridge:
        model.freeze_vl_bridge()
    if training_args.freeze_llm:
        model.freeze_llm()

    train_datasets = build_multi_datasets(
        data_args.dataset_config,
        use_proxy_tokens=training_args.use_proxy_tokens,
        tokenizer=tokenizer,
        img_processor=vis_processor,
    )
    data_collator = DataCollatorForHybridDataset(tokenizer)

    trainer = ComposerTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_datasets,
        data_collator=data_collator
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        raise ValueError("Resume from checkpoint is not supported.")
    else:
        trainer.train()

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    final_checkpoint_dir = f"{training_args.output_dir}/checkpoint-final"
    trainer.save_model(final_checkpoint_dir)
    trainer.save_state()
    vis_processor.save_pretrained(final_checkpoint_dir)


if __name__ == "__main__":
    train()
