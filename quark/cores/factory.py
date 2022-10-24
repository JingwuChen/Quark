# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
"""A global factory to access registered data."""

from quark.cores import registry

_REGISTERED_DATA_CLS = {}
_REGISTERED_MODEL_CLS = {}
_REGISTERED_LOSS_CLS = {}
_REGISTERED_OPTIMIZER_CLS = {}
_REGISTERED_METRICS_CLS = {}
_REGISTERED_TRAIN_CLS = {}
_REGISTERED_PLUGIN_CLS = {}
_REGISTERED_EXP_CLS = {}
_REGISTERED_TASK_CLS = {}


def get_cls(config_cls, dtype):
    if dtype == "data":
        return get_data_cls(config_cls)
    if dtype == "model":
        return get_model_cls(config_cls)
    if dtype == "loss":
        return get_loss_cls(config_cls)
    if dtype == "optimizer":
        return get_optimizer_cls(config_cls)
    if dtype == "metrics":
        return get_metrics_cls(config_cls)
    if dtype == "train":
        return get_train_cls(config_cls)


def register_data_cls(data_config_cls):
    return registry.register(_REGISTERED_DATA_CLS, data_config_cls)


def get_data_cls(data_config_cls):
    return registry.lookup(_REGISTERED_DATA_CLS, data_config_cls)


def register_model_cls(model_config_cls):
    return registry.register(_REGISTERED_MODEL_CLS, model_config_cls)


def get_model_cls(model_config_cls):
    return registry.lookup(_REGISTERED_MODEL_CLS, model_config_cls)


def register_loss_cls(loss_config_cls):
    return registry.register(_REGISTERED_LOSS_CLS, loss_config_cls)


def get_loss_cls(loss_config_cls):
    return registry.lookup(_REGISTERED_LOSS_CLS, loss_config_cls)


def register_optimizer_cls(optimizer_config_cls):
    return registry.register(_REGISTERED_OPTIMIZER_CLS, optimizer_config_cls)


def get_optimizer_cls(optimizer_config_cls):
    return registry.lookup(_REGISTERED_OPTIMIZER_CLS, optimizer_config_cls)


def register_metrics_cls(metrics_config_cls):
    return registry.register(_REGISTERED_METRICS_CLS, metrics_config_cls)


def get_metrics_cls(metrics_config_cls):
    return registry.lookup(_REGISTERED_METRICS_CLS, metrics_config_cls)


def register_train_cls(train_config_cls):
    return registry.register(_REGISTERED_TRAIN_CLS, train_config_cls)


def get_train_cls(train_config_cls):
    return registry.lookup(_REGISTERED_TRAIN_CLS, train_config_cls)


def register_plugin_cls(plugin_config_cls):
    return registry.register(_REGISTERED_PLUGIN_CLS, plugin_config_cls)


def get_plugin_cls(plugin_config_cls):
    return registry.lookup(_REGISTERED_PLUGIN_CLS, plugin_config_cls)


def register_task_cls(task_config_cls):
    return registry.register(_REGISTERED_TASK_CLS, task_config_cls)


def get_task_cls(task_config_cls):
    return registry.lookup(_REGISTERED_TASK_CLS, task_config_cls)


def register_exp_cls(exp_config_cls):
    return registry.register(_REGISTERED_EXP_CLS, exp_config_cls)


def get_exp_cls(exp_config_cls):
    return registry.lookup(_REGISTERED_EXP_CLS, exp_config_cls)
