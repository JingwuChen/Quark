# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
"""A standard Processor to access data config."""
from quark.cores import factory
from quark.cores.node.dag import AbstractProcessor
from quark.cores import get_random_integer
from quark.cores import StandardProcessorOptions
import tensorflow as tf
import abc
from typing import Any, Callable, Optional, List
from absl import logging


@factory.register_task_cls('StandardProcessor')
class StandardProcessor(AbstractProcessor, metaclass=abc.ABCMeta):
    """ returns a tf.data.Dataset"""

    def __init__(self, params: StandardProcessorOptions):
        self._seed = get_random_integer()
        self._global_batch_size = params.global_batch_size
        self._is_training = params.is_training
        self._drop_remainder = params.drop_remainder
        self._shuffle_buffer_size = params.shuffle_buffer_size

        self._cache = params.cache
        self._context = params.context
        self._sharding = params.sharding
        self.plugins = params.plugins

        self._combine_fn = self._get_combine_fn_plugin()
        self._plugins_fn = self._get_plugins_fn_plugin()

    def handler(self, dataset: List[tf.data.Dataset]) -> tf.data.Dataset:

        # 数据shuffle
        _dataset = dataset[0]
        _dataset = tf.nest.map_structure(self._shuffle, _dataset)
        # 若是多个dataset 需要进行conbine
        if tf.nest.is_nested(_dataset):
            if self._combine_fn is None:
                raise ValueError("dataset is mutli, but combine is None")
            _dataset = self._combine_fn(_dataset)

        # 运用plugins
        for _p in self._plugins_fn:
            _dataset = _dataset.apply(_p)

        # share
        if self._context is not None:
            self._read_files_then_shard()

        # batch
        _dataset = _dataset.batch(self._global_batch_size,
                                  drop_remainder=self._drop_remainder)

        # prebatch
        _dataset = _dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return _dataset

    def _maybe_map_fn(
            self,
            ds: tf.data.Dataset,
            fn: Optional[Callable[..., Any]] = None) -> tf.data.Dataset:
        """Calls dataset.map if a valid function is passed in."""
        return ds if fn is None else ds.map(
            fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def _shuffle(self, ds: tf.data.Dataset):
        # If cache is enabled, we will call `shuffle()` later after `cache()`.
        if self._is_training and not self._cache:
            ds = ds.shuffle(self._shuffle_buffer_size, seed=self._seed)
        return ds

    def _get_combine_fn_plugin(self):
        if self.plugins is None:
            return None
        for p in self.plugins:
            if p.startswith("DataPlugin/CombineFn"):
                return factory.get_data_cls(p)
        return None

    def _get_plugins_fn_plugin(self):
        plugins_fn = []
        if self.plugins is None:
            return plugins_fn
        for p in self.plugins:
            if p.startswith("DataPlugin/CombineFn"):
                continue
            plugins_fn.append(factory.get_data_cls(p))
        return plugins_fn

    def _read_files_then_shard(self):
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.OFF)
        self.dataset = self.dataset.with_options(options)
        # Do not enable sharding if tf.data service is enabled, as sharding will be
        # handled inside tf.data service.
        if self._sharding and self._context and (
                self._context.num_input_pipelines > 1):
            self.dataset = self.dataset.shard(
                self._context.num_input_pipelines,
                self._context.input_pipeline_id)


def product_StandardProcessorOptions(conf):
    if conf.get("reuse", False):
        return
    _conf_name = conf.get("conf_name")
    if _conf_name is None:
        raise ValueError("`conf_name` can be specified.")

    _global_batch_size = conf.get("global_batch_size", 2)
    if _global_batch_size is None:
        raise ValueError("`global_batch_size` can be specified.")

    _is_training = conf.get("is_training", False)
    _drop_remainder = conf.get("drop_remainder", False)
    _shuffle_buffer_size = conf.get("shuffle_buffer_size", 100)
    _cache = conf.get("cache", False)

    @factory.register_task_cls(_conf_name)
    class StandardProcessorOptionsNode(StandardProcessorOptions):
        node_name = _conf_name
        global_batch_size: int = _global_batch_size
        is_training: bool = _is_training
        drop_remainder: bool = _drop_remainder
        shuffle_buffer_size: int = _shuffle_buffer_size
        cache: bool = _cache

    logging.info(f"\nregister_data_cls: {_conf_name} is success\n")
