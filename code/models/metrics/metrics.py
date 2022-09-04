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
"""various metric classs"""

import tensorflow as tf
from models.utils.metrics_utils import *

class BatchSoftmaxRecall(tf.keras.metrics.Metric):
    """batch reacall@1 with y_true as diagonal cell matrix
    Args:
        margin: `Float`, margin term in the loss definition.
        Default value is 1.0.
        negative_sample_num:negative sampling num
        reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply.
        Default value is `SUM_OVER_BATCH_SIZE`.
        name: (Optional) name for the loss.
    """
    def __init__(self, name='BatchSoftmaxRecall', **kwargs):
        super(BatchSoftmaxRecall, self).__init__(name=name, **kwargs)
        self.recall = self.add_weight(name=name, initializer='zeros')

    def update_state(self, user_emb, item_emb, sample_weight=None):
        y_true,logits =batch_softmax_logits(user_emb, item_emb)
        values=tf.cast(tf.math.equal(tf.argmax(logits,axis=1),tf.argmax(y_true,axis=1)),tf.float32)
        self.recall.assign(tf.reduce_mean(values))

    def result(self):
        return self.recall

    def reset_state(self):
        self.recall.assign(0.)

class ContrastiveRecall(tf.keras.metrics.Metric):
    """Contrastive loss recall @1 as 0-index as the positive 
    Args:
      negative_sample_num:negative sampling number along batch dimension
      name: (Optional) name for the loss.
    """

    def __init__(self, name='ContrastiveRecall',negative_sample_num=10, **kwargs):
        super(BatchSoftmaxRecall, self).__init__(name=name, **kwargs)
        self.recall = self.add_weight(name=name, initializer='zeros')
        self.negative_sample_num= negative_sample_num

    def update_state(self, user_emb, item_emb, sample_weight=None):
        pos_logits,neg_logits=contrastive_logits(user_emb, item_emb,negative_sample_num=self.negative_sample_num)
        tmp_logits=tf.concat([pos_logits,neg_logits],axis=1)
        values=tf.cast(tf.math.equal(tf.argmax(tmp_logits,axis=1),0),tf.float32)
        self.recall.assign(tf.reduce_mean(values))

    def result(self):
        return self.recall
  
    def reset_state(self):
        self.recall.assign(0.)