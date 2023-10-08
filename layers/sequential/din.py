import tensorflow as tf
from tensorflow import tensordot, expand_dims
from tensorflow.keras import layers,initializers, regularizers, activations, constraints

from .base import DiceMLP
from ..base import MLP


class DinAttention(layers.Layer):
    def __init__(self,activation="dice",**kwargs):
        self.activation=activation
        super(DinAttention,self).__init__(**kwargs)
        
    def build(self,input_shape): ## call 的第一个参数的shape
        """
        query (bs,dim)即候选广告
        key (bs,seqlen,dim)即行为序列
        """
        # if query_shape[-1] != key_shape[-1]:
        #     raise ValueError("query emb_dim % d, expect to be key emb_dim %d" % (query_shape[-1],key_shape[-1]))
        if len(input_shape) != 3:
            raise ValueError("Unexpected inputs dimensions % d, expect to be 3 dimensions" % (len(input_shape)))
        self.maxseqlen=input_shape[1]
        self.emb_dim=input_shape[-1]
        ## 用softmax的话这样
        # # Scale
        # outputs = outputs / (emb_dim ** 0.5)
        # # Activation
        # outputs = tf.nn.softmax(outputs)  # [B, 1, T]

        ## 隐藏层默认relu,activation是最后一层的
        if self.activation.lower()=="dice":
            # hidden_activation = dice
            self.attlayer=DiceMLP(1,[self.emb_dim*2,self.emb_dim],activation="sigmoid")
        else:
            # hidden_activation = relu
            self.attlayer=MLP(1,[self.emb_dim*2,self.emb_dim],activation="sigmoid")
        # self.fc1=layers.Dense(self.emb_dim*2,activation="relu")
        # self.fc2=layers.Dense(self.emb_dim,activation="relu")
        # self.out=layers.Dense(self.emb_dim,activation="sigmoid")
    
    def call(self,sequence,query,seqlen):
        """
        构造数据的时候把行为长度也加为一列
        或直接存mask
        """
        key=sequence
        query=tf.tile(query,[1,self.maxseqlen]) ## (bs,emb_dim) -> (bs,emb_dim*seqlen)
        query=tf.reshape(query,[-1,self.maxseqlen,self.emb_dim]) # (bs,emb_dim*seqlen) -> (bs,seqlen,emb_dim)
        ## key 和 query的一些交互,如果有其它特征other_emb 可以一并拼接
        din_all=tf.concat([query,key,query-key,query*key],axis=-1) ## (bs,seqlen,emb_dim*4)
        ## 一个2层mlp (emb_dim*2,emb_dim,1)
        din_w=self.attlayer(din_all)
        din_w=tf.reshape(din_w,[-1,1,self.maxseqlen])
        ## where 返回bool，或者是满足条件时din_w[i]*paddings[i]
        ## 即din_w[seqlen+1:]=0
        seq_mask=tf.sequence_mask(seqlen, self.maxseqlen, dtype=tf.bool)
        seq_mask=tf.reshape(seq_mask,[-1,1,self.maxseqlen])
        paddings=tf.zeros_like(din_w)
        din_w = tf.where(seq_mask, din_w, paddings)
        ## (bs,1,maxseqlen) @ (bs,maxseqlen,emb_dim) -> (bs,1,emb_dim)
        return tf.reshape(tf.matmul(din_w, key), [-1, self.emb_dim])
        
if __name__=="__main__":
    obj=DinAttention()
    bs=16
    maxseqlen=10
    emb_dim=32
    query=tf.ones((bs,emb_dim))
    key=tf.ones((bs,maxseqlen,emb_dim))
    seqlen=tf.ones((bs))*8
    obj(key,query,seqlen)