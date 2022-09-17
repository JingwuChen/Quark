# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
"""Utility library for picking an appropriate dataset function."""

import functools
from typing import Any, Callable, Type, Union

import tensorflow as tf

PossibleDatasetType = Union[Type[tf.data.Dataset], Callable[[tf.Tensor], Any]]


def pick_dataset_fn(file_type: str) -> PossibleDatasetType:
    if file_type == 'tfrecord':
        return tf.data.TFRecordDataset
    if file_type == 'tfrecord_compressed':
        return functools.partial(tf.data.TFRecordDataset,
                                 compression_type='GZIP')
    raise ValueError('Unrecognized file_type: {}'.format(file_type))
