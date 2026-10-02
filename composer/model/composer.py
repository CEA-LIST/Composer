import json
import copy
import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from typing import List, Optional, Tuple, Union, Dict, Any

from transformers.utils import logging
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import (
    AutoConfig,
    AutoModel,
    PretrainedConfig,
    PreTrainedModel,
    LlamaConfig,
    LlamaModel,
    LlamaForCausalLM,
    CLIPVisionConfig,
    CLIPVisionModel,
    CLIPProcessor
)

from composer.constants import DEFAULT_TOKENS, PROXY_TOKENS, IGNORE_INDEX


logger = logging.get_logger(__name__)


class ComposerConfig(PretrainedConfig):
    model_type = "composer"

    def __init__(
        self,
        llm_cfg=None,
        vision_encoder_cfg=None,
        num_new_token=0,
        vis_output_layer=-2,
        use_proxy_tokens=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if vision_encoder_cfg is None:
            self.vision_encoder_cfg = CLIPVisionConfig()
            logger.info("vision_encoder_cfg is None. initializing the CLIPVisionConfig with default values.")
        elif isinstance(vision_encoder_cfg, dict):
            self.vision_encoder_cfg = CLIPVisionConfig(**vision_encoder_cfg)
        elif isinstance(vision_encoder_cfg, CLIPVisionConfig):
            self.vision_encoder_cfg = vision_encoder_cfg
        else:
            raise NotImplementedError("currently only supports CLIPVisionConfig as vision encoder.")

        if llm_cfg is None:
            self.llm_cfg = LlamaConfig()
            logger.info("llm_cfg is None. Initializing the LlamaModel with default values.")
        elif isinstance(llm_cfg, dict):
            self.llm_cfg = LlamaConfig(**llm_cfg)
        elif isinstance(llm_cfg, LlamaConfig):
            self.llm_cfg = llm_cfg
        else:
            raise NotImplementedError("currently only supports LlamaModel as LLM.")


        self.num_new_token = num_new_token
        self.vocab_size = self.llm_cfg.vocab_size + num_new_token
        self.vis_output_layer = vis_output_layer
        self.use_proxy_tokens = use_proxy_tokens

    def to_json_string(self, use_diff: bool = True) -> str:
        if use_diff:
            config_dict = copy.deepcopy(self)
            config_dict.vision_encoder_cfg = config_dict.vision_encoder_cfg.to_diff_dict()
            config_dict.llm_cfg = config_dict.llm_cfg.to_diff_dict()
            config_dict = config_dict.to_diff_dict()
        else:
            config_dict = copy.deepcopy(self)
            config_dict.vision_encoder_cfg = config_dict.vision_encoder_cfg.to_dict()
            config_dict.llm_cfg = config_dict.llm_cfg.to_dict()
            config_dict = config_dict.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"


class ComposerModel(PreTrainedModel):
    config_class = ComposerConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: ComposerConfig,
        pretrained_vision_encoder: Optional[str] = None,
        pretrained_llm: Optional[str] = None
    ):
        super().__init__(config)
        if pretrained_vision_encoder is None:
            self.vision_encoder = CLIPVisionModel(config.vision_encoder_cfg)
        else:
            self.vision_encoder = CLIPVisionModel.from_pretrained(pretrained_vision_encoder)
            config.vision_encoder_cfg = CLIPVisionConfig.from_pretrained(pretrained_vision_encoder)

        if pretrained_llm is None:
            self.llm = LlamaForCausalLM(config.llm_cfg)
        else:
            self.llm = LlamaForCausalLM.from_pretrained(pretrained_llm)
            config.llm_cfg = LlamaConfig.from_pretrained(pretrained_llm)
            config.vocab_size = config.llm_cfg.vocab_size + config.num_new_token

        image_embed_dim = self.vision_encoder.config.hidden_size
        text_embed_dim = self.llm.config.hidden_size
        self.vl_bridge = nn.Sequential(
            nn.Linear(image_embed_dim, text_embed_dim),
            nn.GELU(),
            nn.Linear(text_embed_dim, text_embed_dim),
        )

        self.extra_lm_head = nn.Linear(text_embed_dim, config.num_new_token, bias=False)

        self.new_input_embs = nn.Embedding(config.num_new_token, text_embed_dim)
        input_embeds = self.llm.get_input_embeddings().weight.data
        input_embeds_avg = input_embeds[:].mean(dim=0, keepdim=True)
        self.new_input_embs.weight.data[:, :] = input_embeds_avg

        self.use_proxy_tokens = config.use_proxy_tokens

        self.pad_token_id = None
        self.img_token_id = None
        self.proxy_token_ids = None
        
        self.post_init()

    def init_special_token_id(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id
        self.img_token_id = tokenizer.convert_tokens_to_ids([DEFAULT_TOKENS['image']])[0]
        if self.use_proxy_tokens:
            self.proxy_token_ids = tokenizer.convert_tokens_to_ids(PROXY_TOKENS)
        return

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlamaModel):
            module.gradient_checkpointing = value

    def freeze_vision_encoder(self):
        self.vision_encoder.requires_grad_(False)

    def freeze_llm(self):
        self.llm.requires_grad_(False)

    def freeze_vl_bridge(self):
        self.vl_bridge.requires_grad_(False)

    def get_vision_encoder(self):
        return getattr(self, 'vision_encoder', None)

    def get_llm(self):
        return getattr(self, 'llm', None)

    def get_input_embeddings(self, input_ids):
        ori_embed_tokens = self.llm.get_input_embeddings()
        mask = input_ids >= ori_embed_tokens.num_embeddings
        ori_ids = input_ids.masked_fill(mask, 0)
        new_ids = input_ids - ori_embed_tokens.num_embeddings
        new_ids = new_ids.masked_fill(~mask, 0)
        input_embeddings = ori_embed_tokens(ori_ids)
        new_input_embeddings = self.new_input_embs(new_ids)
        input_embeddings[mask] = new_input_embeddings[mask]
        return input_embeddings

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache")
        })
        return model_inputs

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache"),
            "images": kwargs.get("images", None)
        })
        
        return model_inputs
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[list] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if past_key_values is None:
            with torch.no_grad():
                # encode image
                vis_encoder_outs = self.vision_encoder(images, output_hidden_states=True)
                select_hidden_state_layer = getattr(self.config, "vis_output_layer", -2)
                image_features = vis_encoder_outs.hidden_states[select_hidden_state_layer][:, 1:]
            
            # if use_proxy_tokens is True, inject N*2 interleaved proxy and image tokens
            # else, inject N interleaved image tokens
            # with N = num_image_tokens
            bs, l, d = image_features.shape

            num_image_tokens = image_features.shape[1]
            new_input_ids = []
            new_labels = []
            for i in range(bs):
                assert self.img_token_id in input_ids[i]
                img_token_pos = (input_ids[i] == self.img_token_id).nonzero(as_tuple=True)[0]
                pad_token_pos = (input_ids[i] == self.pad_token_id).nonzero(as_tuple=True)[0]
                pad_token_pos = pad_token_pos[0] if len(pad_token_pos) > 0 else len(input_ids[i])
                img_placeholder = torch.full((num_image_tokens,), self.img_token_id).to(input_ids.device)
                if self.use_proxy_tokens:
                    proxy_tokens = torch.LongTensor(self.proxy_token_ids[:num_image_tokens]).to(input_ids.device)
                    assert proxy_tokens.shape == img_placeholder.shape
                    img_placeholder = torch.stack([proxy_tokens, img_placeholder], dim=1).reshape(-1)

                new_input_ids.append(torch.cat((
                    input_ids[i][:img_token_pos],
                    img_placeholder,
                    input_ids[i][img_token_pos + 1: pad_token_pos]
                )))
                if labels is not None:
                    new_labels.append(torch.cat((
                        labels[i][:img_token_pos],
                        torch.full((img_placeholder.shape[0],), IGNORE_INDEX, device=labels.device),
                        labels[i][img_token_pos + 1: pad_token_pos]
                    )))
            input_ids = torch.nn.utils.rnn.pad_sequence(
                new_input_ids,
                batch_first=True,
                padding_value=self.pad_token_id)
            if labels is not None:
                labels = torch.nn.utils.rnn.pad_sequence(
                    new_labels,
                    batch_first=True,
                    padding_value=IGNORE_INDEX)
            attention_mask = input_ids.ne(self.pad_token_id)

            # inject visual input
            inputs_embeds = self.get_input_embeddings(input_ids)
            image_features = self.vl_bridge(image_features).to(inputs_embeds.dtype)
            img_mask = input_ids == self.img_token_id
            inputs_embeds.masked_scatter_(img_mask[:, :, None], image_features)

        else:
            bs = past_key_values[0][0].shape[0]
            token_length = past_key_values[0][0].shape[-2] + 1
            attention_mask = torch.ones((bs, token_length), device=attention_mask.device)
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.llm_cfg.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.llm_cfg.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.llm_cfg.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.llm.model(
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.llm.lm_head(hidden_states)
        extra_logits = self.extra_lm_head(hidden_states)
        logits = torch.concat((logits, extra_logits), dim=-1)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]  # + (pred_boxes,)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

AutoConfig.register("composer", ComposerConfig)
AutoModel.register(ComposerConfig, ComposerModel)