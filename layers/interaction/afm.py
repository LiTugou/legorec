import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import (Zeros, glorot_normal, glorot_uniform)
from tensorflow.keras.regularizers import l2


class AFMLayer(layers.Layer):
    def __init__(self,l2_reg=1e-5,dropout=0,**kwargs):
        self.l2_reg=l2_reg
        self.dropout=dropout
        super(AFMLayer,self).__init__(**kwargs)
        
    def build(self,input_shape):
        field_num=input_shape[1]
        emb_size=input_shape[2]
        self.field_num=field_num
        att_factor=field_num*(field_num-1)//2
        
        self.att_W = self.add_weight(shape=(emb_size,
                 att_factor), initializer=glorot_normal(),
                                           regularizer=l2(self.l2_reg), name="attention_W")
        self.att_b = self.add_weight(
            shape=(att_factor,), initializer=Zeros(), name="attention_b")
        self.proj_h = self.add_weight(shape=(att_factor, 1),
                                            initializer=glorot_normal(), name="projection_h")
        self.proj_p = self.add_weight(shape=(
            emb_size, 1), initializer=glorot_normal(), name="projection_p")
        self.dropout = tf.keras.layers.Dropout(
            self.dropout)

    def call(self,inputs):
        veci=[]
        vecj=[]
        for i,j in itertools.combinations(range(self.field_num),2):
            veci.append(tf.expand_dims(inputs[:,i,:],axis=1))
            vecj.append(tf.expand_dims(inputs[:,j,:],axis=1))
        veci=tf.concat(veci,axis=1)
        vecj=tf.concat(vecj,axis=1)
        bits_prod=veci*vecj
        
        #att_tmp=tf.nn.relu(tf.tensordot(bits_prod,self.att_W,axes=(-1,0))+self.att_b) ## (bs,cross_num,att_factor)
        att_tmp=tf.nn.relu(tf.matmul(bits_prod,self.att_W)+self.att_b)
        #att_score=tf.nn.softmax(tf.tensordot(att_tmp,self.proj_h)) ## (bs,cross_num,1)
        att_score=tf.nn.softmax(tf.matmul(att_tmp,self.proj_h))
        att_out=tf.reduce_sum(att_score*bits_prod,axis=1) ## (bs,emb_size)
        att_out=self.dropout(att_out)
        # att_out=tf.tensordot(att_out,self.proj_p) ## (bs,1)
        att_out=tf.matmul(att_out,self.proj_p)
        return att_out
    
    
if __name__=="__main__":
    tmp=tf.zeros((3,6,16))
    obj=AFMLayer()
    obj(tmp)
