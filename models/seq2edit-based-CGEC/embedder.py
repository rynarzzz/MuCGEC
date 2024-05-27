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


class TransformerEmbedder(nn.Module):
    def __init__(self, model_name, tune_bert=False):
        super(TransformerEmbedder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # 根据参数选择是否微调 BERT 模型
        for param in self.model.parameters():
            param.requires_grad = bool(tune_bert)

        # 存储隐藏层大小
        self.hidden_size = self.model.config.hidden_size

    def forward(self, text_list):
        """
        将文本列表转换为嵌入表示。
        :param text_list: 文本列表
        :return: 最后一层的隐藏状态
        """
        # 使用 tokenizer 处理输入文本
        try:
            inputs = self.tokenizer(text_list, return_tensors='pt', padding=True, truncation=True)
        except ValueError:
            print(text_list)
            raise

        # 转移到相同的设备上
        inputs = {name: tensor.to(self.model.device) for name, tensor in inputs.items()}

        # 获取模型输出
        outputs = self.model(**inputs)

        # 返回最后一层的隐藏状态
        return outputs.last_hidden_state

    def get_output_dim(self):
        """
        获取输出层的维度。
        :return: Transformer模型的隐藏层大小
        """
        return self.hidden_size


# 使用示例
def get_transformer_embedder(model_name, tune_bert=False):
    return TransformerEmbedder(model_name, tune_bert)