#coding:utf-8
import tensorflow as tf
from tensorflow.keras import layers,initializers, regularizers, activations, constraints


class SELayer(layers.Layer):
    def __init__(self,hidden_units,**kwargs):
        super(SELayer,self).__init__(**kwargs)
        self.hidden_units=hidden_units
        
    def build(self,input_shape):
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d, expect to be 3 dimensions" % (len(input_shape)))
        self.dense_layer=[]
        for unit in self.hidden_units:
            self.dense_layer.append(layers.Dense(unit,use_bias=False,activation="relu"))
        self.outlayer=layers.Dense(input_shape[1],use_bias=False,activation="relu")
        
    def call(self,inputs):
        """
        inputs: (bs,field_num,emb_size)
        output: (bs,field_num,emb_size)
        """
        weight=tf.reduce_mean(inputs,axis=-1)
        for dense in self.dense_layer:
            weight=dense(weight)
        weight=self.outlayer(weight)
        ## (bs,field_num) -->> (bs,field_num,1)
        weight=tf.expand_dims(weight,-1)
        return inputs*weight