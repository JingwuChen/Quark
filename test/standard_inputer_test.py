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
from quark.cores import standard_inputer


@factory.register_data_cls('Test_FeatureConfig_user')
class TestFeatureUserConfig(cfg.FeatuerConfig):
    name = "user"
    len = 1
    value_dtype = 3


@factory.register_data_cls('Test_FeatureConfig_item')
class TestFeatureItemConfig(cfg.FeatuerConfig):
    name = "item"
    len = 1
    value_dtype = 3


@factory.register_data_cls('Test_FeatureConfig_ctr')
class TestFeatureCtrConfig(cfg.FeatuerConfig):
    name = "ctr"
    dtype = 1
    len = 1
    value_dtype = 1


@factory.register_data_cls('Test_news_rec')
class TestStandardConfig(cfg.DataConfig):
    tfds_name = "Test"
    mode = "tfrecord"
    files = "../example/data/mini_news.test.tfrecord"
    buffer_size = 100
    num_parallel_reads = 2
    feature_config = [
        factory.get_data_cls('Test_FeatureConfig_user'),
        factory.get_data_cls('Test_FeatureConfig_item'),
        factory.get_data_cls('Test_FeatureConfig_ctr')
    ]


data_config = factory.get_data_cls('Test_news_rec')

sta = factory.get_data_cls('StandardInputer')(data_config)
ds = sta.read()

i = 1
for d in ds:
    i += 1
    if i > 2:
        break
    print(d)
