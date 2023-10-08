import tensorflow.compat.v2 as tf

from tensorflow.keras import layers,activations,backend,constraints,initializers,regularizers

class CrossLayer(layers.Layer):
    def __init__(
        self,
        layer_nums,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        self.layer_nums=layer_nums
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self,input_shape):
        ## units = field_num*emb_size+dense_num
        units=input_shape[-1]
        dtype = tf.as_dtype(self.dtype or backend.floatx())
        self.kernel = self.add_weight(
            "kernel",
            shape=[self.layer_nums,units,1],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )
        self.bias = self.add_weight(
            "bias",
            shape=[self.layer_nums,units],
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            dtype=self.dtype,
            trainable=True,
        )

    def call(self,inputs):
        x0=inputs
        xl=inputs
        for i in range(self.layer_nums):
            xl_w=tf.matmul(xl,self.kernel[i,:,:]) ## (bs,1)
            xl=x0*xl_w+xl+self.bias[i,:]
        return xl
    
if __name__=="__main__":
    a=tf.zeros((64,128))
    obj=CrossLayer(2)
    obj(a)
