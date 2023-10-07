#coding:utf-8
import tensorflow as tf
from tensorflow.keras import layers, initializers, regularizers, activations, constraints
import tensorflow.keras.backend as K
    
    
class MLP(layers.Layer):
    def __init__(self,
                 out_dim,
                 hidden_units,
                 hidden_activation="relu",
                 activation=None,
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs
        ):
        super().__init__(**kwargs)
        dense_layers=[]
        for unit in hidden_units:
            dense=layers.Dense(unit,activation=hidden_activation,
                               use_bias=use_bias,
                               kernel_initializer=kernel_initializer,
                               bias_initializer=bias_initializer,
                               kernel_regularizer=kernel_regularizer,
                               bias_regularizer=bias_regularizer,
                               activity_regularizer=activity_regularizer,
                               kernel_constraint=kernel_constraint,
                               bias_constraint=bias_constraint
                              )
            dense_layers.append(dense)
        self.outlayer=layers.Dense(out_dim,activation=activation,  
                               use_bias=use_bias,
                               kernel_initializer=kernel_initializer,
                               bias_initializer=bias_initializer,
                               kernel_regularizer=kernel_regularizer,
                               bias_regularizer=bias_regularizer,
                               activity_regularizer=activity_regularizer,
                               kernel_constraint=kernel_constraint,
                               bias_constraint=bias_constraint
                              )
        self.dense_layers=dense_layers
        
    def call(self,inputs):
        for dense in self.dense_layers:
            inputs=dense(inputs)
        return self.outlayer(inputs)

