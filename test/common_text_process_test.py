# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

"""A standard inputer to access data config."""
import sys

sys.path.append("..")
print("==================")
print(sys.path)
print("==================")
from quark.cores import factory
from quark.cores import config_definitions as cfg
from quark.cores import common_text_processer


# @factory.register_process_cls('Test_ProcessConfig')
# class TestProcessConfig(cfg.DataConfig):
#     global_batch_size = 32


@factory.register_process_cls('Test_news_rec')
class TestStandardConfig(cfg.DataConfig):
    global_batch_size = 32
    # feature_config = [
    #     factory.get_process_cls('Test_ProcessConfig'),
    # ]


data_config = factory.get_process_cls('Test_news_rec')
sta = factory.get_process_cls('CommonTextProcessor')(data_config)

string = ['我是中国人']
output = sta.handler(string)
print(output)
