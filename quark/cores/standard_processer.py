# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

"""A standard inputer to access data config."""
from quark.cores import factory
from quark.cores import AbstractProcessor
from quark.cores import get_random_integer, is_dir, is_file, get_files_from_dir
from quark.cores import config_definitions as cfg
from typing import List
import tensorflow as tf
import abc


@factory.register_process_cls('StandardProcessor')
class StandardProcessor(AbstractProcessor, metaclass=abc.ABCMeta):
    """ returns a tf.data.Dataset"""
    static_randnum = get_random_integer()

    def __init__(self, params: cfg.DataConfig):
        self.global_batch_size = params.global_batch_size

    def handler(self):
        print("batch_size: ", self.global_batch_size)
