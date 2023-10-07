import itertools
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers, initializers, regularizers, activations, constraints

class BilinearInteraction(layers.Layer):
    """BilinearInteraction Layer used in FiBiNET.
      Input: (batch_size,1,embedding_size).
      Output: (batch_size,1,embedding_size)
      bilinear_type:
          1. interaction wi,j
          2. each wi
          3. all w
    """

    def __init__(self, bilinear_type="interaction",
                 kernel_initializer="glorot_normal",
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 **kwargs
        ):
        self.bilinear_type = bilinear_type
        super().__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        
    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('dimention of inputs should be (bs,field_num,embedding_size)')
        self.field_num=input_shape[1]
        embedding_size = input_shape[-1] ## K

        if self.bilinear_type == "all":
         ## Field-All Type: W_list 的 shape 为 K * K
            self.W = self.add_weight(shape=(embedding_size, embedding_size), 
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint,
                                     name="bilinear_weight")
        elif self.bilinear_type == "each":
         ## Field-Each Type: W 的 shape 为 F * K * K
            # self.W_list = [self.add_weight(shape=(embedding_size, embedding_size), 
            #        initializer=glorot_normal(), name="bilinear_weight" + str(i)) for i in range(len(input_shape) - 1)]
            self.W = self.add_weight(shape=(self.field_num - 1,embedding_size, embedding_size), 
                             initializer=self.kernel_initializer,
                             regularizer=self.kernel_regularizer,
                             constraint=self.kernel_constraint,
                             name="bilinear_weight")
        elif self.bilinear_type == "interaction":
         ## Field-Interaction Type: W_list 的 shape 为 F*(F - 1)/2 * K * K
            self.W_list = [self.add_weight(shape=(embedding_size, embedding_size), 
                                           initializer=self.kernel_initializer,
                                           regularizer=self.kernel_regularizer,
                                           constraint=self.kernel_constraint, name="bilinear_weight" + str(i) + '_' + str(j)) for i, j in
                           itertools.combinations(range(self.field_num), 2)]


    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (K.ndim(inputs)))
  
        ## 下面的计算用 einsum避免一些for（没事找事）
        if self.bilinear_type == "all":
            vidots = tf.matmul(inputs,self.W)
            p = [tf.multiply(vidots[:,i,:], inputs[:,j,:]) for i, j in itertools.combinations(range(self.field_num), 2)]
        elif self.bilinear_type == "each":
            # vidots = [tf.tensordot(inputs[i], self.W_list[i], axes=(-1, 0)) for i in range(n - 1)]
            # p = [tf.multiply(vidots[i], inputs[j]) for i, j in itertools.combinations(range(n), 2)]
            vidots = tf.einsum("ijk,jkk->ijk",inputs[:,:-1,:],self.W)
            p = [tf.multiply(vidots[:,i,:], inputs[:,j,:]) for i, j in itertools.combinations(range(self.field_num), 2)]
        elif self.bilinear_type == "interaction":
            p = [tf.multiply(tf.matmul(inputs[:,i,:],w), inputs[:,j,:])
                 for (i,j), w in zip(itertools.combinations(range(self.field_num), 2), self.W_list)]
        else:
            raise NotImplementedError
        return tf.stack(p,axis=1)
    

    
if __name__ == "__main__":
    inputs=tf.ones((64,23,16))
    obj=BilinearInteraction()
    obj(inputs)