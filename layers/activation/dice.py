#coding:utf-8
import tensorflow as tf
from tensorflow.keras import layers

class Dice(layers.Layer):
    """
    https://zhuanlan.zhihu.com/p/78829402
    """
    def __init__(self,epsilon=0.000000001,axis=-1,**kwargs):
        self.axis=axis
        self.epsilon=epsilon
        
        ## 加kwargs是为了trainable参数的初始设置， 创建实例后设置dice的trainable也会自动设置bnlayer的
        self.bnlayer=layers.BatchNormalization(
                            axis=axis,
                            epsilon=self.epsilon,
                            center=False,
                            scale=False,
                            **kwargs
        )
        super().__init__(**kwargs)
    
    def build(self,input_shape):
        alphas=tf.compat.v1.get_variable('alpha_dice', input_shape[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        shape=[1 for _ in range(len(input_shape)-1)]+[input_shape[-1]]
        self.alphas=tf.reshape(alphas,shape)

    def call(self, _x):
        inputs_normed = self.bnlayer(_x)
        x_p = tf.sigmoid(inputs_normed)
        return self.alphas * (1.0 - x_p) * _x + x_p * _x