# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
"""A standard inputer to access data config."""
import tensorflow as tf
import abc
from absl import logging
from quark.cores import factory
from quark.cores import PrintConfig
from quark.cores import PrintDataSetOptions
from typing import List


@factory.register_task_cls('PrintDataSet')
class PrintDataSet(metaclass=abc.ABCMeta):

    def __init__(self, params: PrintConfig):
        self.read_num = params.read_num

    def handler(self, dataset: List[tf.data.Dataset]) -> tf.data.Dataset:
        _dataset = dataset[0]
        n = 1
        for d in _dataset:
            logging.info(f"\n {d} \n")
            n += 1
            if n > self.read_num:
                break
        return _dataset


def product_PrintDataSetOptions(conf):
    if conf.get("reuse", False):
        return
    _conf_name = conf.get("conf_name")
    _read_num = conf.get("read_num", 2)

    @factory.register_task_cls(_conf_name)
    class PrintDataSetOptionsNode(PrintDataSetOptions):
        node_name = _conf_name
        read_num: int = _read_num

    logging.info(f"\nregister_data_cls: {_conf_name} is success\n")
