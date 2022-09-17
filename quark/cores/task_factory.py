# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
"""A global factory to register and access all registered tasks."""

from quark.cores import registry

_REGISTERED_TASK_CLS = {}


def register_task_cls(task_config_cls):
    return registry.register(_REGISTERED_TASK_CLS, task_config_cls)


def get_task(task_config, **kwargs):
    if task_config.BUILDER is not None:
        return task_config.BUILDER(task_config, **kwargs)
    return get_task_cls(task_config.__class__)(task_config, **kwargs)


def get_task_cls(task_config_cls):
    task_cls = registry.lookup(_REGISTERED_TASK_CLS, task_config_cls)
    return task_cls
