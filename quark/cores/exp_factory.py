# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
"""Experiment factory methods."""
from quark.cores import registry

_REGISTERED_CONFIGS = {}


def register_config_factory(name):
    """Register ExperimentConfig factory method."""
    return registry.register(_REGISTERED_CONFIGS, name)


def get_exp_config(exp_name: str):
    """Looks up the `ExperimentConfig` according to the `exp_name`."""
    exp_creater = registry.lookup(_REGISTERED_CONFIGS, exp_name)
    return exp_creater()