# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
"""A global factory to access NLP registered data loaders."""

from quark.cores import registry

_REGISTERED_DATA_LOADER_CLS = {}


def register_data_loader_cls(data_config_cls):
    return registry.register(_REGISTERED_DATA_LOADER_CLS, data_config_cls)


def get_data_loader(data_config):
    """Creates a data_loader from data_config."""
    return registry.lookup(_REGISTERED_DATA_LOADER_CLS,
                           data_config.__class__)(data_config)
