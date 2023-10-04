#coding:utf-8
import tensorflow as tf
from tensorflow import tensordot, expand_dims
from tensorflow.keras import layers, Model, initializers, regularizers, activations, constraints, Input, Sequential
import tensorflow.keras.backend as K

class MLP(layers.Layer):
    def __init__(self,out_dim,hidden_units,activation=None):
        super(MLP,self).__init__()
        dense_layers=[]
        for unit in hidden_units:
            dense=layers.Dense(unit,activation=tf.nn.relu)
            dense_layers.append(dense)
        self.outlayer=layers.Dense(out_dim,activation=activation)
        self.dense_layers=dense_layers
        
    def call(self,inputs):
        for dense in self.dense_layers:
            inputs=dense(inputs)
        return self.outlayer(inputs)

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