from layers.sequential.base import ActivationUnit,AttentionPoolingLayer
from layers.sequential._dien.grulayer import AUGRUCell
from layers.base import MLP
import tensorflow as tf
from tensorflow.keras import layers,activations,backend,constraints,initializers,regularizers



# https://blog.csdn.net/qq_42363032/article/details/122365548
class InterestExtractLayer(layers.Layer):
    def __init__(self, extract_units, extract_dropout=0,**kwargs):
        self.extract_dropout=extract_dropout
        self.extract_units=extract_units
        super().__init__(**kwargs)
        # 用一个mlp来计算 auxiliary loss
    
    def build(self,input_shape):
        emb_size=input_shape[-1]
        # 传统的GRU来抽取时序行为的兴趣表示  return_sequences=True: 返回上次的输出
        self.auxiliary_mlp = MLP(1,self.extract_units,dropout_rate=self.extract_dropout)
        self.rnn = layers.GRU(units=emb_size, activation='tanh', recurrent_activation='sigmoid', return_sequences=True)

    def call(self, user_behavior, mask_bool, neg_user_behavior=None, neg_mask_bool=None):
        """
            user_behavior : (2000, 40, 4)
            mask : (2000, 40, 1)
            neg_user_behavior : (2000, 39, 4)
            neg_mask : (2000, 39, 1)
        """
        # 将0-1遮罩变换bool
        # mask_bool = tf.cast(tf.squeeze(mask, axis=2), tf.bool)  # (2000, 40)

        gru_interests = self.rnn(user_behavior, mask=mask_bool)  # (2000, 40, 4)

        # 计算Auxiliary Loss，只在负采样的时候计算 aux loss
        if neg_user_behavior is not None:
            # 此处用户真实行为user_behavior为图中的e，GRU抽取的状态为图中的h
            gru_embed = gru_interests[:, 1:]  # (2000, 39, 4)
            #neg_mask_bool = tf.cast(tf.squeeze(neg_mask, axis=2), tf.bool)  # (2000, 39)

            # 正样本的构建  选取下一个行为作为正样本
            pos_seq = tf.concat([gru_embed, user_behavior[:, 1:]], -1)  # (2000, 39, 8)
            pos_res = self.auxiliary_mlp(pos_seq)  # (2000, 39, 1)
            pos_res = tf.sigmoid(pos_res[neg_mask_bool])  # 选择不为0的进行sigmoid  (N, 1) ex: (18290, 1)
            pos_target = tf.ones_like(pos_res, tf.float16)  # label

            # 负样本的构建  从未点击的样本中选取一个作为负样本
            neg_seq = tf.concat([gru_embed, neg_user_behavior], -1)  # (2000, 39, 8)
            neg_res = self.auxiliary_mlp(neg_seq)  # (2000, 39, 1)
            neg_res = tf.sigmoid(neg_res[neg_mask_bool])
            neg_target = tf.zeros_like(neg_res, tf.float16)

            # 计算辅助损失 二分类交叉熵
            aux_loss = tf.keras.losses.binary_crossentropy(tf.concat([pos_res, neg_res], axis=0), tf.concat([pos_target, neg_target], axis=0))
            aux_loss = tf.cast(aux_loss, tf.float32)
            aux_loss = tf.reduce_mean(aux_loss)

            return gru_interests, aux_loss

        return gru_interests, 0
    


class InterestEvolutionLayer(layers.Layer):
    def __init__(self,
                 attention_units=[32,16],
                 attention_dropout=0,
                 gru_type="augru",
                 **kwargs):
        self.gru_type=gru_type
        self.attention_units=attention_units
        self.attention_dropout=attention_dropout
        super().__init__(**kwargs)
        
    def build(self,input_shape):
        emb_size=input_shape[-1]
        self.maxseqlen=input_shape[1]
        if self.gru_type.upper() == "AUGRU":
            self.attention = AttentionPoolingLayer(units=self.attention_units,
                                                   dropout_rate=self.attention_dropout,
                                                   return_score=True)
            self.rnn=layers.RNN(AUGRUCell(units=emb_size))
        elif self.gru_type.upper() == "GRU":
            self.attention = AttentionPoolingLayer(dropout=self.attention_dropout, units=self.attention_units)
            self.rnn=layers.GRU(units=emb_size,return_sequences=True)
            
        elif self.gru_type.upper() in ("AIGRU") :
            self.attention = AttentionPoolingLayer(dropout=self.attention_dropout, units=self.attention_units)
            self.rnn=layers.GRU(units=emb_size)
        else:
            raise NotImplementedError

    def call(self,gru_interests,query_ad,mask_bool):
        if self.gru_type.upper() == 'GRU':
            # GRU后接attention
            out = self.rnn(gru_interests, mask=mask_bool)  # (2000, 40, 4)
            out = self.attention(out, query_ad, mask_bool)  # (2000, 40, 4)
            out = tf.reduce_sum(out, axis=1)  # (2000, 4)
        elif self.gru_type.upper() == 'AIGRU':
            # AIGRU
            att_score = self.attention(gru_interests, query_ad, mask_bool)  # (2000, 40, 1)
            out = att_score * gru_interests  # (2000, 40, 4)
            out = self.rnn(out, mask=mask_bool)  # (2000, 4)
        elif self.gru_type.upper() == 'AUGRU':
            # AGRU or AUGRU
            att_score = self.attention(gru_interests, query_ad,  mask_bool)  # (2000, 40, 1)
            out = self.rnn((gru_interests, att_score), mask=mask_bool)  # (2000, 4)
        else:
            raise NotImplementedError
        return out
    

    
class DienLayer(layers.Layer):
    def __init__(self,extract_units,attention_units,extract_dropout=0.2,attention_dropout=0.2,gru_type="AUGRU",**kwargs):
        self.attention_dropout=attention_dropout
        self.attention_units=attention_units
        self.extract_dropout=extract_dropout
        self.extract_units=extract_units
        self.gru_type=gru_type
        super().__init__(**kwargs)
        self.extract=InterestExtractLayer(
            extract_units=self.extract_units,
            extract_dropout=self.extract_dropout
        )
        self.evolu=InterestEvolutionLayer(
            attention_units=self.attention_units,
            attention_dropout=self.attention_dropout,
            gru_type=self.gru_type
        )

    def build(self,input_shape):
        self.maxseqlen=input_shape[1]
        
    def call(self, user_behavior, query_ad, seqlen , neg_user_behavior=None, neg_seqlen=None):
        mask_bool=tf.sequence_mask(seqlen,self.maxseqlen)
        if neg_seqlen is not None:
            neg_mask_bool=tf.sequence_mask(neg_seqlen,self.maxseqlen)
        else:
            neg_mask_bool=None
        gru_interests,auxu_loss=self.extract(user_behavior, mask_bool , neg_user_behavior=neg_user_behavior, neg_mask_bool=neg_mask_bool)
        final_interest=self.evolu(gru_interests,query_ad,mask_bool)
        return final_interest
    
if __name__ == "__main__":    
    obj=DienLayer([128,128],[128,128])
    user_behavior=tf.ones((20, 40, 4))
    query_ad=tf.ones((20, 4))
    seqlen=tf.ones((20,))*30
    obj(user_behavior, query_ad, seqlen)