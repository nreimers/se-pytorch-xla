import torch
from transformers import (AutoModel)
from torch import nn
import torch_xla.core.xla_model as xm
import os


class AutoModelForSentenceEmbedding(nn.Module):
    def __init__(self, model_name, tokenizer, normalize=True, pooling='mean'):
        super(AutoModelForSentenceEmbedding, self).__init__()

        self.model = AutoModel.from_pretrained(model_name)
        self.normalize = normalize
        self.tokenizer = tokenizer
        self.pooling = pooling

    def forward(self, **kwargs):
        model_output = self.model(**kwargs)
        if self.pooling == 'mean':
            embeddings = self.mean_pooling(model_output, kwargs['attention_mask'])
        elif self.pooling == 'cls':
            embeddings = self.cls_pooling(model_output, kwargs['attention_mask'])

        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def cls_pooling(self, model_output, attention_mask):
        return model_output[0][:, 0]

    def save_pretrained(self, output_path):
        if xm.is_master_ordinal():
            self.tokenizer.save_pretrained(output_path)
            self.model.config.save_pretrained(output_path)

        xm.save(self.model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
