#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：MuCGEC 
@File    ：embedder.py
@IDE     ：PyCharm 
@Author  ：rengengchen
@Time    ：2024/5/24 16:40 
"""
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from utils.mismatched_utils import MisMatchedEmbedder


class SeqEncoder(nn.Module):
    def __init__(self, sub_token_mode, encoder_path, device, tune_bert=False):
        super().__init__()
        self.matched_embedder = AutoModel.from_pretrained(encoder_path)
        self.hidden_size = self.matched_embedder.config.hidden_size
        self.mismatched_embedder = MisMatchedEmbedder(device, sub_token_mode)
        self.activate_grad = True
        self.tune_bert = tune_bert

    def forward(self, input_dict):
        requires_grad = bool(self.tune_bert)
        if self.activate_grad ^ requires_grad:
            for param in self.parameters():
                param.requires_grad_(requires_grad)
            self.activate_grad = requires_grad

        output_dict = self.matched_embedder(
            input_ids=input_dict["input_ids"],
            token_type_ids=input_dict["token_type_ids"],
            attention_mask=input_dict["attention_mask"],
        )
        last_hidden_states = output_dict[0]
        word_embeddings = self.mismatched_embedder.get_mismatched_embeddings(
            last_hidden_states,
            offsets=input_dict["offsets"],
            word_mask=input_dict["word_mask"])
        return word_embeddings

    def get_output_dim(self):
        """
        获取输出层的维度。
        :return: Transformer模型的隐藏层大小
        """
        return self.hidden_size


# 使用示例
def get_transformer_embedder(model_name, tune_bert=False):
    return SeqEncoder(model_name, tune_bert)
