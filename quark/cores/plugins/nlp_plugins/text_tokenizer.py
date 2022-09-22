#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author: GuidoHwang
@time  : 2022/09/03
@Des   :
"""
import unicodedata
from typing import List, Union, Optional, Tuple
from transformers import BertTokenizer

if __name__ == '__main__':
    text = "我爱北京天安门，吢吣"
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    tokens = tokenizer.tokenize(text)
    print("tokens: ", tokens)

    tokens_input_ids = tokenizer(tokens, max_length=64, truncation=True,add_special_tokens=False).input_ids
    print(tokens_input_ids)

    pass
