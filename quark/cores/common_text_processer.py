# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

"""A standard inputer to access data config."""
from quark.cores import factory
from quark.cores import AbstractProcessor
from quark.cores import get_random_integer, is_dir, is_file, get_files_from_dir
from quark.cores import config_definitions as cfg
from quark.cores.plugins.nlp_plugins import text_normalize, text_segment
from typing import List
import tensorflow as tf
import abc


@factory.register_process_cls('CommonTextProcessor')
class CommonTextProcessor(AbstractProcessor, metaclass=abc.ABCMeta):
    """ returns a tf.data.Dataset"""
    static_randnum = get_random_integer()

    def __init__(self, params: cfg.DataConfig):
        self.global_batch_size = params.global_batch_size

    def handler(self, text_list: List[str]) -> List[list]:
        '''
        :return: 文本token id序列
        '''
        print("文本归一化......")
        print('输入为：', text_list)
        output = []
        for text in text_list:
            text_half = text_normalize.full2half(text)
            text_low_case = text_normalize.upper2lower(text_half)
            text_simple = text_normalize.traditional2simplified(text_low_case)
            text_seg= text_segment.segment(text_simple)
            output.append(text_seg)

        return output
