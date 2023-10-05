#coding:utf-8
import tensorflow as tf
from tensorflow.keras import layers, initializers, regularizers, activations, constraints, Sequential
import tensorflow.keras.backend as K

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
    
    
class MLP(layers.Layer):
    def __init__(self,out_dim,hidden_units,hidden_activation="relu",activation=None,**kwargs):
        super(MLP,self).__init__(**kwargs)
        dense_layers=[]
        for unit in hidden_units:
            dense=layers.Dense(unit,activation=hidden_activation)
            dense_layers.append(dense)
        self.outlayer=layers.Dense(out_dim,activation=activation)
        self.dense_layers=dense_layers
        
    def call(self,inputs):
        for dense in self.dense_layers:
            inputs=dense(inputs)
        return self.outlayer(inputs)

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
        weight=tf.reduce_mean(inputs,axis=-1)
        for dense in self.dense_layer:
            weight=dense(weight)
        weight=self.outlayer(weight)
        ## (bs,field_num) -->> (bs,field_num,1)
        weight=tf.expand_dims(weight,-1)
        return inputs*weight
    
class FMCrossLayer(layers.Layer):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self, **kwargs):

        super(FMLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        ## (bs,feat_len,emb_dim)
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d, expect to be 3 dimensions" % (len(input_shape)))

        super(FMLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, **kwargs):
        """
        缺少线性层，需要hash之后的bucket
        """
        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))
        concated_embeds_value = inputs
        # 先求和再平方
        square_of_sum = tf.square(tf.reduce_sum(concated_embeds_value, axis=1, keepdims=True))
        # 先平方再求和
        sum_of_square = tf.reduce_sum(concated_embeds_value * concated_embeds_value, axis=1, keepdims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keepdims=False)

        return cross_term