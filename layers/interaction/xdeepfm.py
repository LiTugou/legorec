#coding:utf-8
import tensorflow as tf
from tensorflow.keras import layers, initializers, regularizers, activations, constraints


class CinLayer(layers.Layer):
    def __init__(self, cin_size,**kwargs):
        super().__init__(**kwargs)
        self.cin_size = cin_size  # 每层的矩阵个数

    def build(self, input_shape):
        # input_shape: [None, n, k]
        self.field_num = [input_shape[1]] + self.cin_size  # 每层的矩阵个数(包括第0层)

        # field_num[i+1]个形状为self.field_num[0]*self.field_num[i]的矩阵
        # 用于压缩三维矩阵得到field_num[i+1]个向量
        self.cin_W = [self.add_weight(  
                         name='Cin_w'+str(i),
                         shape=(1, self.field_num[0]*self.field_num[i], self.field_num[i+1]),
                         initializer=tf.initializers.glorot_uniform(),
                         regularizer=tf.keras.regularizers.l1_l2(1e-5),
                         trainable=True)
                      for i in range(len(self.field_num)-1)]

    def call(self, inputs):
        # inputs: [None, n, k]
        k = inputs.shape[-1]
        res_list = [inputs]
        X0 = tf.split(inputs, k, axis=-1)           # 最后维切成k份，list: k * [None, n, 1]
        for i, size in enumerate(self.field_num[1:]):
            Xi = tf.split(res_list[-1], k, axis=-1) # list: k * [None, field_num[i], 1]
            x = tf.matmul(X0, Xi, transpose_b=True) # tensor: [ k, None, field_num[0], field_num[i]]
            x = tf.reshape(x, (k, -1, self.field_num[0]*self.field_num[i]))
                                                    # [k, None, field_num[0]*field_num[i]]
            x = tf.transpose(x, [1, 0, 2])          # [None, k, field_num[0]*field_num[i]]
            x = tf.nn.conv1d(input=x, filters=self.cin_W[i], stride=1, padding='VALID')  # 用卷积实现三维矩阵的压缩
                                                    # (None, k, field_num[i+1])
            x = tf.transpose(x, [0, 2, 1])          # (None, field_num[i+1], k)
            res_list.append(x)

        res_list = res_list[1:]   # 去掉 X0
        res = tf.concat(res_list, axis=1)     # (None, field_num[1]+...+field_num[n], k)
        output = tf.reduce_sum(res, axis=-1)  # (None, field_num[1]+...+field_num[n])
        return output