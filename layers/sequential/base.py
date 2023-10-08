from tensorflow.keras import layers
import tensorflow as tf
from ..activation.dice import Dice
from ..base import MLP

# mini-batch aware regularization 还未实现
# https://github.com/zhougr1993/DeepInterestNetwork/issues/82
class DiceMLP(layers.Layer):
    def __init__(self,out_dim,hidden_units,dropout_rate=0,activation=None,**kwargs):
        super(DiceMLP,self).__init__(**kwargs)
        dense_layers=[]
        ac_layers=[]
        for unit in hidden_units:
            dense=layers.Dense(unit)
            ac=Dice()
            dense_layers.append(dense)
            ac_layers.append(ac)
            if dropout_rate>1e-8:
                ac_layers.append(layers.Dropout(rate=dropout_rate))
        self.outlayer=layers.Dense(out_dim,activation=activation)
        self.dense_layers=dense_layers
        self.ac_layers=ac_layers
        
    def call(self,inputs):
        for i in range(len(self.dense_layers)):
            inputs=self.dense_layers[i](inputs)
            inputs=self.ac_layers[i](inputs)
        return self.outlayer(inputs)
    

class ActivationUnit(layers.Layer):

    def __init__(self, units=[32, 16], dropout_rate=0.2):
        super().__init__()
        self.dicemlp = DiceMLP(1,units,dropout_rate=dropout_rate)

    def build(self,input_shape):
        self.emb_size=input_shape[2]
        self.maxseqlen=input_shape[1]
        
    def call(self, sequence, query):
        """
            query : 单独的ad的embedding mat -> batch * 1 * embed
            user_behavior : 行为特征矩阵 -> batch * seq_len * embed
        """
        key=sequence
        query=tf.tile(query,[1,self.maxseqlen]) ## (bs,emb_dim) -> (bs,emb_dim*seqlen)
        query=tf.reshape(query,[-1,self.maxseqlen,self.emb_size]) # (bs,emb_dim*seqlen) -> (bs,seqlen,emb_dim)
        ## key 和 query的一些交互,如果有其它特征other_emb 可以一并拼接
        din_all=tf.concat([query,key,query-key,query*key],axis=-1) ## (bs,seqlen,emb_dim*4)
        ## 一个2层mlp (emb_dim*2,emb_dim,1)
        din_w=self.dicemlp(din_all)
        return din_w

'''   Attention Pooling Layer   '''
class AttentionPoolingLayer(layers.Layer):

    def __init__(self,units=[32, 16], dropout_rate=0.2, return_score=False):
        super().__init__()
        self.active_unit = ActivationUnit(units,dropout_rate)
        self.return_score = return_score

    def call(self,user_behavior,query, mask_bool):
        """
            query_ad : 单独的ad的embedding mat -> batch * 1 * embed
            user_behavior : 行为特征矩阵 -> batch * seq_len * embed
            mask : 被padding为0的行为置为false -> batch * seq_len * 1
        """

        # attn weights
        attn_weights = self.active_unit(user_behavior,query)
        # mul weights and sum pooling
        if self.return_score:
            output = user_behavior * attn_weights *tf.expand_dims(tf.cast(mask_bool,tf.float32),axis=-1)
            return output

        return attn_weights
    
if __name__ == "__main__":
    sequence=tf.ones((64,20,16))
    query=tf.ones((64,16))
    mask_bool=tf.sequence_mask(tf.ones((64,))*10,20)
    obj=ActivationUnit()
    obj(sequence,query)
    obj=AttentionPoolingLayer(return_score=True)
    tmp=obj(sequence,query,mask_bool)
