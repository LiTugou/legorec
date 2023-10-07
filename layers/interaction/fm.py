import tensorflow as tf
from tensorflow.keras import layers, initializers, regularizers, activations, constraints
import tensorflow.keras.backend as K


class FMCrossLayer(layers.Layer):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
      Input: (batch_size,field_size,embedding_size).
      Output: (batch_size, 1).
    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        
    def build(self, input_shape):
        ## (bs,feat_len,emb_dim)
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d, expect to be 3 dimensions" % (len(input_shape)))


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