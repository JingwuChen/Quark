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
"""mlp related layers"""

from tensorflow.python.ops.math_ops import MatMul
from tensorflow.python.ops.nn_ops import BiasAdd
import tensorflow as tf
from tensorflow.python.keras import activations, initializers
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.layers import Dropout
from typing import List,Optional,Union,Tuple
import random




class MLPLayer(tf.keras.layers.Layer):
    """a more powerful mlp layer
    Args:
        units:  liear layer units optional list type of integer .
        activation:activation function name ,default None
        use_bias: (Optional) whether to use bias.
        is_batch_norm: (Optional) whether to use batch norm.
        drop_rate: (Optional) dropout rate.
        kernel_initializer: variable weight initializers name for kernel.
        bias_initializer:bias initializers name
    """

    def __init__(self,
                 units:Union[List[int],int],
                 activation:Optional[str]=None,
                 use_bias:Optional[bool]=True,
                 is_batch_norm:Optional[bool]=False,
                 drop_rate:float=0,
                 kernel_initializer:str='glorot_uniform',
                 bias_initializer:str='zeros',
                 **kwargs):
        super(MLPLayer, self).__init__(**kwargs)

        self.units = [units] if not isinstance(units, list) else units
        if len(self.units) <= 0:
            raise ValueError(
                f'Received an invalid value for `units`, expected '
                f'a positive integer, got {units}.')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.is_batch_norm = is_batch_norm
        self.drop_rate = drop_rate
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')

        dims = [last_dim] + self.units

        self.kernels = []
        self.biases = []
        self.bns = []
        for i in range(len(dims) - 1):
            self.kernels.append(
                self.add_weight(f'kernel_{i}',
                                shape=[dims[i], dims[i + 1]],
                                initializer=self.kernel_initializer,
                                trainable=True))

            if self.use_bias:
                self.biases.append(
                    self.add_weight(f'bias_{i}',
                                    shape=[
                                        dims[i + 1],
                                    ],
                                    initializer=self.bias_initializer,
                                    trainable=True))
            self.bns.append(tf.keras.layers.BatchNormalization())
        self.built = True

    def call(self, inputs, training=False):
        _input = inputs
        for i in range(len(self.units)):
            _input = MatMul(a=_input, b=self.kernels[i])
            if self.use_bias:
                _input = BiasAdd(value=_input, bias=self.biases[i])
            # BN
            if self.is_batch_norm:
                _input = self.bns[i](_input)
            # ACT
            if self.activation is not None:
                _input = self.activation(_input)
            # DROP
            if self.drop_rate > 0:
                _input = Dropout(self.drop_rate)(_input,training=training)

        return _input

class MultiMLPLayer(tf.keras.layers.Layer):
    """a multiple mlp layer
    Args:
        fc_size:  `List of Int` multiple mlp units.
        fc_activation:`List of string or None` multiple mlp activation function name
        fc_drop_prob: `List of float` multiple mlp drop out rate
    """

    def __init__(self,fc_size:List[int],fc_activation:List[Optional[str]],fc_drop_prob:List[float]):
        super(MultiMLPLayer, self).__init__()
        self.fc_drop_prob=fc_drop_prob
        self.fc_size = fc_size
        self.fc_activation= fc_activation

    def build(self,input_shape=None):
        self.fc_layers=tf.keras.Sequential()
        for size,activation_function,drop_prob in zip(self.fc_size,self.fc_activation,self.fc_drop_prob):
            self.fc_layers.add(MLPLayer(units=[size], activation=activation_function,drop_rate=drop_prob))

    def call(self,x, training=False):
        return self.fc_layers(x,training=training)


class TowerLayer(tf.keras.layers.Layer):
    """single tower layer for dssm
    Args:
        fc_size:  `List of Int` multiple mlp units.
        fc_activation:`List of string or None` multiple mlp activation function name
        fc_drop_prob: `List of float` multiple mlp drop out rate
        name: `String` name of tower output embedding tensor
    Return:
        tower output embedding tensor,shape of [D,emb_size]    
    
    """
    def __init__(self,fc_size:List[int],fc_activation:List[Optional[str]],
                    fc_drop_prob:List[float],name=None,*args,**kwargs):
        super(TowerLayer,self).__init__(*args,**kwargs)
        self.fc_drop_prob=fc_drop_prob
        self.fc_size = fc_size
        self.fc_activation= fc_activation
        self.out_emb_name = name if name else "out"+str(random.randint(1,2))
        self.args= args
        self.kwargs= kwargs

    def build(self, input_shape=None):
        self.fc_layers=MultiMLPLayer(fc_size=self.fc_size,fc_activation=self.fc_activation,fc_drop_prob=self.fc_drop_prob)

    def call(self,input_emb:List[tf.Tensor],training=False):
        input_emb = tf.keras.layers.Flatten()(tf.concat(input_emb, axis=1))
        input_emb=self.fc_layers(input_emb,training=training)
        input_emb=tf.math.l2_normalize(input_emb,axis=1,name=self.name)
        return input_emb

class MaskLayer(tf.keras.layers.Layer):
    """whether mask inputs along sequence dimension"""
    def __init__(self,*args,**kwargs):
        super(MaskLayer, self).__init__(*args,**kwargs)

    def call(self,inputs,emb_inputs):
        mask_inputs= tf.math.logical_not(tf.math.equal(inputs, 0))#shape :(batch,channel)
        mask_inputs = tf.cast(mask_inputs, dtype=emb_inputs.dtype)
        out=tf.reduce_mean(emb_inputs*tf.expand_dims(mask_inputs,axis=-1),axis=1)
        return out




class PiontwiseFFN(tf.keras.layers.Layer):
    def __init__(self,d_model=512,dff=1024,*args,**kwargs):
        super(PiontwiseFFN, self).__init__(*args,**kwargs)
        self.ffn=tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # Shape `(batch_size, seq_len, dff)`.
      tf.keras.layers.Dense(d_model)  # Shape `(batch_size, seq_len, d_model)`.
  ])

    def call(self,x):
        return self.ffn(x)
