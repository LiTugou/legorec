#coding:utf-8
import tensorflow as tf
from tensorflow.keras import layers, initializers, regularizers, activations, constraints
import tensorflow.keras.backend as K
    
    
class MLP(layers.Layer):
    def __init__(self,
                 out_dim,
                 hidden_units,
                 dropout_rate=0,
                 use_bn=False,
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
            if use_bn:
                tmp=layers.BatchNormalization(axis=-1)
                dense_layers.append(tmp)
            if dropout_rate>1e-8:
                tmp=layers.Dropout(rate=dropout_rate)
                dense_layers.append(tmp)
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


class MultiHeadAttention(layers.Layer):
    def __init__(self, num_heads, **kwargs):
        self.num_heads=num_heads
        super().__init__(**kwargs)

    def build(self,input_shape):
        self.emb_size=input_shape[-1]
        self.head_size = model_size // num_heads
        self.WQ = keras.layers.Dense(emb_size, name="dense_query")
        self.WK = keras.layers.Dense(emb_size, name="dense_key")
        self.WV = keras.layers.Dense(emb_size, name="dense_value")
        self.dense = keras.layers.Dense(emb_size)
        
    def call(self, query, key, value, mask):
        # query: (batch, maxlen, emb_size)
        # key  : (batch, maxlen, emb_size)
        # value: (batch, maxlen, emb_size)
        batch_size = tf.shape(query)[0]

        # shape: (batch, maxlen, model_size)
        query = self.WQ(query)
        key = self.WK(key)
        value = self.WV(value)

        def _split_heads(x):
            x = tf.reshape(x, shape=[batch_size, -1, self.num_heads, self.head_size])
            return tf.transpose(x, perm=[0, 2, 1, 3])

        # shape: (batch, num_heads, maxlen, head_size)
        query = _split_heads(query)
        key = _split_heads(key)
        value = _split_heads(value)

        # shape: (batch, num_heads, maxlen, maxlen)
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        # 缩放 matmul_qk
        dk = tf.cast(query.shape[-1], tf.float32)
        score = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            # mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype=tf.float32)
            score += (1 - mask) * -1e9

        alpha = tf.nn.softmax(score)
        context = tf.matmul(alpha, value)
        context = tf.transpose(context, perm=[0, 2, 1, 3])
        context = tf.reshape(context, (batch_size, -1, self.model_size))
        output = self.dense(context)
            
        return output