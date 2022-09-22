#!/usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author: GuidoHwang
@time  : 2022/09/03
@Des   : 文本切分工具
"""
import jieba


def segment(sentence: str) -> list:
    return list(jieba.cut(sentence))


if __name__ == '__main__':
    string = '我是中国人，我爱我的祖国！'
    print(segment(string))
    pass
