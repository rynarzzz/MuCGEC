#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：MuCGEC 
@File    ：vocab.py
@IDE     ：PyCharm 
@Author  ：rengengchen
@Time    ：2024/5/24 16:55 
"""


class Vocabulary:
    def __init__(self):
        self._token_to_index = {}
        self._index_to_token = {}
        self._namespaces = set()

    def add_token(self, token, namespace='tokens'):
        """添加一个token到指定的namespace。"""
        if namespace not in self._token_to_index:
            self._token_to_index[namespace] = {}
            self._index_to_token[namespace] = {}

        # 只有当token不在当前namespace的词汇表中时才添加
        if token not in self._token_to_index[namespace]:
            index = len(self._token_to_index[namespace])
            self._token_to_index[namespace][token] = index
            self._index_to_token[namespace][index] = token
            self._namespaces.add(namespace)

    def get_token_index(self, token, namespace='tokens'):
        """从指定namespace获取token的索引。如果不存在，则返回-1。"""
        return self._token_to_index.get(namespace, {}).get(token, -1)

    def get_token_from_index(self, index, namespace='tokens'):
        """从指定namespace通过索引获取token。如果不存在，则返回None。"""
        return self._index_to_token.get(namespace, {}).get(index, None)

    def get_vocab_size(self, namespace='tokens'):
        """获取指定namespace的词汇表大小。"""
        return len(self._token_to_index.get(namespace, {}))
