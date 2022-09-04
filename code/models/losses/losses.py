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
"""various losses classs"""

import tensorflow as tf
from models.losses.base_losses import LossFunctionWrapper
from models.utils.losses_utils import *

class ContrastiveLoss(LossFunctionWrapper):
    """
    Args:
      margin: `Float`, margin term in the loss definition.
        Default value is 1.0.
      negative_sample_num:negative sampling num
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply.
        Default value is `SUM_OVER_BATCH_SIZE`.
      name: (Optional) name for the loss.
    """

    def __init__(
        self,
        margin: float = 1.0,
        negative_sample_num=10,
        reduction: str = tf.keras.losses.Reduction.SUM,
        name: str = "contrastive_loss",
    ):
        super().__init__(
            contrastive_loss, reduction=reduction, name=name, margin=margin,
            negative_sample_num=negative_sample_num
        )

class BatchSoftmaxLoss(LossFunctionWrapper):
    """
    Args:
      tau: tempreature para.
        Default value is 20.
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply.
        Default value is `SUM_OVER_BATCH_SIZE`.
      name: (Optional) name for the loss.
    """

    def __init__(
        self,
        tau: float = 20,
        reduction: str = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        name: str = "batch_softmax_loss",
    ):
        super().__init__(
            batch_softmax_loss, reduction=reduction, name=name, tau=tau
        )
