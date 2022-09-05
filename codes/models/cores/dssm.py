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
"""dssm model"""

import tensorflow as tf
from typing import List, Union, Optional, Tuple
from models.layers.mlp import TowerLayer, MaskLayer, Dict
import numpy as np


class DSSM(tf.keras.Model):
    def __init__(self,

                 fc_size: List[int],
                 fc_activation: List[Optional[str]],
                 fc_drop_prob: List[float],
                 vocal_size: int = 10000,  # Number of decoder layers.
                 embed_size: int = 32,  # Input/output dimensionality.
                 emb_init: Optional[np.ndarray] = None,
                 mask_emb=True,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.mask_emb = mask_emb
        if emb_init is not None:
            self.emb = tf.keras.layers.Embedding(vocal_size, embed_size,
                                                 embeddings_initializer=tf.keras.initializers.Constant(emb_init))
        else:
            self.emb = tf.keras.layers.Embedding(vocal_size,
                                                 embed_size,
                                                 embeddings_regularizer="l2")
        self.user_embed_tower = TowerLayer(fc_size=fc_size, fc_activation=fc_activation, fc_drop_prob=fc_drop_prob,
                                           name='query_out_emb')
        self.item_embed_tower = TowerLayer(fc_size=fc_size, fc_activation=fc_activation, fc_drop_prob=fc_drop_prob,
                                           name='item_out_emb')

    def call(self, input_dic: Dict[str, tf.Tensor], training):
        # 输入
        # 结构
        emb_dic = {}
        for k, v in input_dic.items():
            tmp = self.emb(v)
            if self.mask_emb:
                tmp = MaskLayer()(v, tmp)
            emb_dic[k] = tmp
        user_emb = []
        item_emb = []
        for k, v in emb_dic.items():
            if k in self.user_feature_name:
                user_emb.append(v)
            if k in self.item_feature_name:
                item_emb.append(v)
        user_emb = self.user_embed_tower(user_emb)
        item_emb = self.item_embed_tower(item_emb)
        return user_emb, item_emb
