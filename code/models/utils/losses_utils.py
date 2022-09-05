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
"""various losses function"""
import tensorflow as tf
from models.utils.metrics_utils import contrastive_logits,batch_softmax_logits

@tf.function
def contrastive_loss(
    user_emb: tf.Tensor, item_emb: tf.Tensor, margin: float = 1.0,negative_sample_num=10
    ) -> tf.Tensor:
    r"""Computes the contrastive loss between `user_emb` and `item-emb`.
    This loss encourages the embedding to be close to each other for
    the samples of the same label and the embedding to be far apart at least
    by the margin constant for the samples of different labels.
    Args:
      user_emb: [D,m] integer `Tensor` with shape `[batch_size,emb_size]` of
        user embedding of encoder.
      item_emb: [D,m] float `Tensor` with shape `[batch_size,emb_size]` of
        item embedding of encoder.
      margin: margin term in the loss definition.
      negative_sample_num:negative sampling num

    Returns:
      contrastive_loss: 1-D float `Tensor` with shape `[batch_size,negative_sample_num]`.
    """
    pos_logits,neg_logits=contrastive_logits(user_emb, item_emb,negative_sample_num=negative_sample_num)
    return tf.maximum(0.0,margin-(pos_logits-neg_logits))

@tf.function
def batch_softmax_loss(
    user_emb: tf.Tensor, item_emb: tf.Tensor, tau: float = 20
) -> tf.Tensor:
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
      cross entropy loss: 1-D float `Tensor` with shape `[batch_size,batch_size]`.
    """
    y_true,logits =batch_softmax_logits(user_emb, item_emb)
    y_true=tf.cast(y_true, dtype=logits.dtype)
    # return -y_true*tf.math.log(tf.nn.softmax(logits*tau,axis=1))
    return tf.nn.softmax_cross_entropy_with_logits(y_true,logits*tau)