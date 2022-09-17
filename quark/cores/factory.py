# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
"""A global factory to access registered data."""

from quark.cores import registry

_REGISTERED_DATA_CLS = {}
_REGISTERED_EXP_CLS = {}
_REGISTERED_TASK_CLS = {}


# 对于基于配置文件产生的 DataConfig和Class 注册到这
def register_data_cls(data_config_cls):
    return registry.register(_REGISTERED_DATA_CLS, data_config_cls)


# 通过注册的名字取出你的 DataConfig和Class 配置信息
def get_data_cls(data_config_cls):
    return registry.lookup(_REGISTERED_DATA_CLS)(data_config_cls)


# 对于基于配置文件产生的 TaskConfig和Class 注册到这
def register_task_cls(task_config_cls):
    return registry.register(_REGISTERED_TASK_CLS, task_config_cls)


# 通过注册的名字取出你的 TaskConfig和Class 配置信息
def get_task_cls(task_config_cls):
    return registry.lookup(_REGISTERED_TASK_CLS, task_config_cls)


# 对于基于配置文件产生的 ExperimentConfig和Class 注册到这
def register_exp_cls(exp_config_cls):
    return registry.register(_REGISTERED_EXP_CLS, exp_config_cls)


# 通过注册的名字取出你的 ExperimentConfig和Class 配置信息
def get_exp_config(exp_config_cls):
    return registry.lookup(_REGISTERED_EXP_CLS, exp_config_cls)