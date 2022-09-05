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
"""function to calculate metrics"""

import tensorflow as tf
from typing import List,Tuple
from models.utils.function_utils import generate_negative_sample_index

def contrastive_logits(
    user_emb: tf.Tensor, item_emb: tf.Tensor, negative_sample_num=10
    ) -> Tuple[tf.Tensor]:
    r"""Computes the contrastive loss between `user_emb` and `item-emb`.
    This loss encourages the embedding to be close to each other for
    the samples of the same label and the embedding to be far apart at least
    by the margin constant for the samples of different labels.
    Args:
      user_emb: [D,m] integer `Tensor` with shape `[batch_size,emb_size]` of
        user embedding of encoder.
      item_emb: [D,m] float `Tensor` with shape `[batch_size,emb_size]` of
        item embedding of encoder.
      negative_sample_num:negative sampling num

    Returns:
      contrastive logits: [D,1] float `Tensor` with shape `[batch_size,1] of pos_logits,
                [D,k] float `Tensor` with shape [batch_size,negative_sample_num] of negative logits`.
    """
    sample_index=generate_negative_sample_index(item_emb,negative_sample_num)#(batch,negative_sample_num)
    sample_item_emb=tf.gather(item_emb,sample_index)#shape :(batch,negative_sample_num,emb_size)
    neg_logits=tf.squeeze(tf.matmul(sample_item_emb,tf.expand_dims(user_emb,axis=-1)),axis=2)#shape :(batch,negative_sample_num)
    pos_logits=tf.reduce_sum(item_emb*user_emb,axis=1,keepdims=True)#shape:(batch,1)
    return pos_logits,neg_logits

def batch_softmax_logits(
    user_emb: tf.Tensor, item_emb: tf.Tensor
) -> Tuple[tf.Tensor]:
    r"""Computes the softmax loss between `y_true` and `y_pred`.
    This loss encourages the embedding to be close to each other for
    the samples of the same label and the embedding to be far apart
    for the other samples.
    Args:
      user_emb: [D,m] integer `Tensor` with shape `[batch_size,emb_size]` of
        user embedding of encoder.
      item_emb: [D,m] float `Tensor` with shape `[batch_size,emb_size]` of
        item embedding of encoder.
      tau: tempreature para.
    Returns:
      
      contrastive logits: [D,D] float `Tensor` with shape `[batch_size,1] of pos_logits,
                [D,D] float `Tensor` with shape [batch_size,negative_sample_num] of negative logits`.
    """
    logits= tf.matmul(user_emb,tf.transpose(item_emb))#[batch,batch]
    y_true=tf.linalg.band_part(tf.ones_like(logits),0,0)
    return y_true,logits