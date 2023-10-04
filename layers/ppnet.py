#coding:utf-8
import tensorflow as tf
from tensorflow import tensordot, expand_dims
from tensorflow.keras import layers, Model, initializers, regularizers, activations, constraints, Input, Sequential

from tensorflow.keras.backend import expand_dims,repeat_elements,sum

class PPGate(layers.Layer):
    def __init__(self, 
                 output_dim,
                 hidden_dim,
                 activation="ReLU",
                 dropout_rate=0.0,
                 batch_norm=False,
                 scope=None
                ):
        """
        hidden_dim ppgate的隐藏层(一般1层完事)
        output_dim 应该等于feature_emb field*field_emb
        """
        super(PPGate, self).__init__()
        
        gatelayer=tf.keras.Sequential()
        for pid, pp_unit in enumerate(hidden_dim):
            if scope is not None:
                name=f'pp_gate_{scope}_{pid}'
            else:
                name=f'pp_gate_{pid}'
            gate=layers.Dense(pp_unit,activation=tf.nn.relu,name=name)
            if batch_norm:
                bn=layers.BatchNormalization()
                gatelayer.add(bn)
            gatelayer.add(gate)
        if scope is not None:
            name=f'pp_gate_out_{scope}_{pid}'
        else:
            name=f'pp_gate_out_{pid}'
        out=layers.Dense(output_dim,activation=tf.nn.sigmoid,
                            bias_initializer=tf.ones_initializer(),
                            name=name)
        gatelayer.add(out)
        self.gatelayer=gatelayer
        

    def call(self, inputs):
        """
        sg_net = tf.stop_gradient(other_emb)
        inputs = tf.concat([sg_emb, bias_emb], axis=1)
        """
        return self.gatelayer(inputs) * 2
    
class PPNetBlock(layers.Layer):
    def __init__(self,
                 output_dim,
                 gate_hidden_dim,
                 hidden_units,
                 activation="ReLU",
                 dropout_rates=0.0,
                 batch_norm=False,
                 use_bias=True):
        """
        input_dim： 是feature_emb的维度,第一个gate的输出形状
        output_dim： mlp的输出
        gate_hidden_dim： gate网络的隐藏层
        hidden_units: mlp的隐藏层
        
        每一个dense层的输出都会过一个gate
        """
        super(PPNetBlock, self).__init__()
#         if not isinstance(dropout_rates, list):
#             dropout_rates = [dropout_rates] * len(hidden_units)

#         hidden_activations = [get_activation(x) for x in hidden_activations]
        self.output_dim=output_dim
        self.hidden_units=hidden_units
        self.gate_hidden_dim=gate_hidden_dim
        self.hidden_units=hidden_units
        self.activation=activation
        self.dropout_rates=dropout_rates
        self.use_bias=use_bias
        self.batch_norm=batch_norm
        
    
    def build(self, input_shape):
        gate_layers=[]
        dense_layers=[]
        bn_layers=[]
        hidden_units=[input_shape[-1]]+self.hidden_units
        
        for i in range(len(hidden_units)-1):
            gate=PPGate(hidden_units[i],self.gate_hidden_dim,self.activation)
            dense=layers.Dense(hidden_units[i+1],activation=tf.nn.relu)
            if self.batch_norm:
                bn=layers.BatchNormalization()
                bn_layers.append(bn)
            gate_layers.append(gate)
            dense_layers.append(dense)
            
        self.out=tf.keras.Sequential()
        self.out.add(layers.BatchNormalization())
        self.out.add(layers.Dense(self.output_dim))
        
        self.gate_layers=gate_layers
        self.bn_layers=bn_layers
        self.dense_layers=dense_layers
        
    def call(self, feature_emb, gate_emb):
        sg_emb = tf.stop_gradient(feature_emb)
        gate_input = tf.concat([sg_emb, gate_emb], axis=1)
        
        hidden = feature_emb
        for i in range(len(self.dense_layers)):
            gw = self.gate_layers[i](gate_input)
            if self.batch_norm:
                hidden = self.bn_layers[i](hidden * gw)
                hidden = self.mlp_layers[i](hidden)
            else:
                hidden = self.dense_layers[i](hidden * gw)
        return self.out(hidden)
    
if __name__ == "__main__":
    gate=PPGate(128,[256])
    ppnet=PPNetBlock(1,[256],[256,128,64])

    bs=16
    gate(tf.zeros([bs,64]))

    # test_inputs=tf.zeros([bs,5120])
    # bias_emb=tf.zeros([bs,64])
    # ppnet(test_inputs,bias_emb)