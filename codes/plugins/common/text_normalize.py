#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author: GuidoHwang
@time  : 2022/09/03
@Des   : 文本标准化工具
"""
import zhconv


def full_to_half(sentence: str) -> str:
    # 全角转半角
    change_sentence = ""
    for word in sentence:
        inside_code = ord(word)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        change_sentence += chr(inside_code)
    return change_sentence


def upper2lower(sentence: str) -> str:
    # 大写字母转为小写字母
    new_sentence = sentence.lower()
    return new_sentence


def traditional2simplified(sentence: str) -> str:
    # 转为简体
    sentence = zhconv.convert(sentence, 'zh-hans')  # 转简体
    return sentence


if __name__ == '__main__':
    text = '我是中国人Chinese，我爱我的祖国！'
    # text_normalize = TextNormalize()
    # final = text_normalize.normalize_utf8(text)
    final = full_to_half(text)
    print(final)
    final = upper2lower(text)
    print(final)
    final = traditional2simplified('綠樹陰濃夏日長，樓臺倒影入池塘。')
    print(final)
