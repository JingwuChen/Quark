# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function

"""A standard inputer to access data config."""
from quark.cores import factory
from quark.cores import AbstractInputer
from quark.cores import get_random_integer, is_dir, is_file, get_files_from_dir
from quark.cores import config_definitions as cfg
from typing import List
import tensorflow as tf
import abc


@factory.register_data_cls('StandardInputer')
class StandardInputer(AbstractInputer, metaclass=abc.ABCMeta):
    """ returns a tf.data.Dataset"""
    static_randnum = get_random_integer()

    def __init__(self, params: cfg.DataConfig):

        # file
        if params.files is None:
            raise ValueError('`files` can be specified, but got %s.' %
                             params.files)
        if isinstance(params.files, list):
            self.files = params.files
        elif isinstance(params.files, str):
            if is_file(params.files):
                self.files = [params.files]
            elif is_dir(params.files):
                self.files = get_files_from_dir(params.files)
        # mode
        if params.mode is None:
            raise ValueError('`mode` can be specified, but got %s.' %
                             params.mode)
        self.mode = params.mode
        self.compression_type = params.compression_type
        self.buffer_size = params.buffer_size
        self.num_parallel_reads = params.num_parallel_reads
        # feature config
        self.feature_config = params.feature_config
        if self.feature_config is None or len(self.feature_config) <= 0:
            raise ValueError('`feature_config` can be specified, but got %s.' %
                             self.feature_config)
        # textline config
        self.textline_split = params.textline_split

    def read(self) -> tf.data.Dataset:
        if self.mode == "tfrecord":
            return self.read_TFRecord()

        if self.mode == "textline":
            return self.read_TextLine()

    def read_TFRecord(self) -> tf.data.Dataset:
        if self.mode != "tfrecord":
            raise ValueError('`mode` is not tfrecord')
        # step1
        self.ds = tf.data.TFRecordDataset(self.files)
        self.ds = self.ds.map(self._parse_sample_tfrecord,
                              self.num_parallel_reads)
        return self.ds

    def read_TextLine(self) -> tf.data.Dataset:
        if self.mode != "textline":
            raise ValueError('`mode` is not textline')
        self.ds = tf.data.TextLineDataset(self.files)
        self.ds = self.ds.map(self._parse_sample_textline,
                              self.num_parallel_reads)
        return self.ds

    def _parse_sample_tfrecord(self, example):
        features = dict()
        labels = dict()
        for feat_cfg in self.feature_config:
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
