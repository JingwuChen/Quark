# Copyright 2022 The UniNLP Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility to retrieve function args."""

import tensorflow as tf
from typing import List,Optional,Union,Any
import numpy as np

def is_tensor_or_variable(x:Union[tf.Tensor,tf.Variable,Any])->bool:
    r"""determine whether x is a tensor or variable.
    """
    return tf.is_tensor(x) or isinstance(x, tf.Variable)

def generate_negative_sample_index(input_array:tf.Tensor,negative_sample_num:int
        )->tf.Tensor:
    r"""generate a negative randomly sampling index along the batch dimmension
    Args:
      input_array: [D,m] embedding `Tensor` with shape `[batch_size,emb_size]` of
        last output.
      item_emb: [D,m] float `Tensor` with shape `[batch_size,emb_size]` of
        item embedding of encoder.
      negative_sample_num:sampling number as negative in batch

    Returns:
      sampling index: [D,negative_sample_num] sampling index,
                
    """
    batch_num=tf.shape(input_array)[0]
    ta = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    for i in tf.range(batch_num):
        #make random index not equal to self index
        tmp_array=tf.random.uniform(shape=(negative_sample_num,), minval=i+1, maxval=i+batch_num, dtype=tf.int32)%batch_num
        # tmp_array=np.random.randint(i+1,i+batch_num,size=negative_sample_num) % batch_num
        ta=ta.write(i,tmp_array)
    return ta.stack()


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates
  

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # Apply the sine function to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # Apply the cosine function to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)