import torch
import transformers
from dataclasses import dataclass

from composer.constants import IGNORE_INDEX


@dataclass
class DataCollatorForHybridDataset(object):

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        meta_keys = ('input_ids', 'labels', 'image', 'source')
        input_ids, labels, images, sources = tuple(
            [instance.get(key, None) for instance in instances] for key in meta_keys)
      
        if all([x is not None for x in images]):
            images = torch.stack(images)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            images=images,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        )
        return batch