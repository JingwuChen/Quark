# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
"""A standard inputer to access data config."""
from quark.cores import factory
from quark.cores.node.dag import AbstractInputer
from quark.cores import is_dir, is_file, get_files_from_dir
from quark.cores import StandardInputerOptions
from quark.cores import FieldOptions
import tensorflow as tf
import abc
from absl import logging
from typing import Union, List


@factory.register_task_cls('StandardInputer')
class StandardInputer(AbstractInputer, metaclass=abc.ABCMeta):
    """ returns a tf.data.Dataset"""

    def __init__(self, params: StandardInputerOptions):

        # file
        self._files = params.files
        if self._files is None:
            raise ValueError('`files` can be specified.')
        if isinstance(params.files, str):
            if is_file(params.files):
                self._files = [params.files]
            elif is_dir(params.files):
                self._files = get_files_from_dir(params.files)
        # mode
        self._mode = params.mode
        if self._mode is None:
            raise ValueError('`mode` can be specified.')
        self._compression_type = params.compression_type
        self._buffer_size = params.buffer_size
        self._num_parallel_reads = params.num_parallel_reads
        # feature config
        self._feature_config = params.feature_config
        if self._feature_config is None or len(self._feature_config) <= 0:
            raise ValueError('`feature_config` can be specified.')
        # textline config
        self._textline_split = params.textline_split

    def handler(self) -> tf.data.Dataset:
        if self._mode == "tfrecord":
            return self.read_TFRecord()

        if self._mode == "textline":
            return self.read_TextLine()

    def read_TFRecord(self) -> tf.data.Dataset:
        self.ds = tf.data.TFRecordDataset(self._files)
        self.ds = self.ds.map(self._parse_sample_tfrecord,
                              self._num_parallel_reads)
        return self.ds

    def read_TextLine(self) -> tf.data.Dataset:
        self.ds = tf.data.TextLineDataset(self._files)
        self.ds = self.ds.map(self._parse_sample_textline,
                              self._num_parallel_reads)
        return self.ds

    def _parse_sample_tfrecord(self, example):
        features = dict()
        labels = dict()
        for feat_cfg in self._feature_config:
            name = feat_cfg.name
            dtype = feat_cfg.dtype
            len = feat_cfg.len
            value_dtype = feat_cfg.value_dtype
            if value_dtype == 1:
                value_dtype = tf.float32
            elif value_dtype == 2:
                value_dtype = tf.string
            elif value_dtype == 3:
                value_dtype = tf.int64
            if dtype == 0:
                features[name] = tf.io.FixedLenFeature(shape=[len],
                                                       dtype=value_dtype)

            elif dtype == 1:
                labels[name] = tf.io.FixedLenFeature(shape=[len],
                                                     dtype=value_dtype)

        features = tf.io.parse_single_example(example, features)

        if labels:
            labels = tf.io.parse_single_example(example, labels)
            return features, labels
        return features

    def _parse_sample_textline(self, sample):
        pass


def product_StandardInputerOptions(conf):
    if conf.get("reuse", False):
        return
    _conf_name = conf.get("conf_name", None)
    if _conf_name is None:
        raise ValueError("`conf_name` can be specified.")

    _mode = conf.get("mode", None)
    if _mode is None:
        raise ValueError("`mode` can be specified.")

    _files = conf.get("files", None)
    if _files is None:
        raise ValueError("`files` can be specified.")

    _field_config = conf.get("field_config", None)
    if _field_config is None:
        raise ValueError("`field_config` can be specified.")

    _compression_type = conf.get("compression_type", None)
    _buffer_size = conf.get("buffer_size", 100)
    _num_parallel_reads = conf.get("num_parallel_reads",
                                   tf.data.experimental.AUTOTUNE)

    _field_config_list = []
    for feat in _field_config:
        product_FieldOptions(feat)

    for feat in _field_config:
        feat_node_name = feat["conf_name"]
        _field_config_list.append(factory.get_data_cls(feat_node_name))

    @factory.register_task_cls(_conf_name)
    class StandardInputerOptionsNode(StandardInputerOptions):
        conf_name = _conf_name
        mode: str = _mode
        files: Union[List[str], str] = _files
        compression_type: str = _compression_type
        buffer_size: int = _buffer_size
        num_parallel_reads: int = _num_parallel_reads
        feature_config: List[FieldOptions] = _field_config_list

    logging.info(f"\nregister_data_cls: {_conf_name} is success\n")


def product_FieldOptions(conf):
    if conf.get("reuse", False):
        return
    _conf_name = conf.get("conf_name")
    if _conf_name is None:
        raise ValueError("`conf_name` can be specified.")
    _name = conf.get("name")
    if _name is None:
        raise ValueError("`name` can be specified.")

    _dtype = conf.get("dtype", 0)
    _len = conf.get("len", 1)
    _value_dtype = conf.get("value_dtype", 1)

    @factory.register_data_cls(_conf_name)
    class FieldOptionsNode(FieldOptions):
        conf_name = _conf_name
        name = _name
        dtype: int = _dtype
        len: int = _len
        value_dtype: int = _value_dtype

    logging.info(f"\n register_data_cls: {_conf_name} is success \n")
