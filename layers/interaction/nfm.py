import tensorflow as tf
from tensorflow.keras import layers,activations,backend,constraints,initializers,regularizers

# https://datawhalechina.github.io/fun-rec/#/ch02/ch2.2/ch2.2.3/NFM
class BiInteraction(layers.Layer):
    def __init__(self,
                 dropout_rate=0.2,
                 **kwargs):
        self.dropout_rate=dropout_rate
        super().__init__(**kwargs)
        self.bnlayer=layers.BatchNormalization(name='bi_interaction_bn')
        self.dropoutlayer=layers.Dropout(rate=dropout_rate)

    def call(self,inputs):
        """
        inputs: (bs,field_num,emb_size)
        output: (bs,emb_size)
        """
        sum_square_part = tf.square(tf.reduce_sum(inputs, axis=1)) # (batch, emb_size)
        square_sum_part = tf.reduce_sum(tf.square(inputs), axis=1) # (batch, emb_size)
        nfm = 0.5 * (sum_square_part - square_sum_part)
        nfm = self.bnlayer(nfm)
        nfm = self.dropoutlayer(nfm)
        return nfm

if __name__ == "__main__":
    obj=NFM()
    a=tf.ones((64,23,16))
    obj(a)