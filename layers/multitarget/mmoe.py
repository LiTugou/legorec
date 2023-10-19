#coding:utf-8
import tensorflow as tf
from tensorflow.keras import layers,initializers, regularizers, activations, constraints
from tensorflow.keras.backend import expand_dims,repeat_elements,sum

from ..base import MLP
class MMoELayer(layers.Layer):
    """
    Multi-gate Mixture-of-Experts model.
    expert_units 专家网络MLP结构，expert_units[:-1]是隐藏层，expert_units[-1]为output
    gate_units ， gate_units是gate的隐藏层，num_expert是output
    有多少 task 就有几个 gate
    """
    def __init__(self,
                 num_experts,
                 expert_units,
                 num_tasks,
                 gate_units,
                 use_expert_bias=True,
                 use_gate_bias=True,
                 expert_activation='relu',
                 gate_activation='softmax',
                 expert_bias_initializer='zeros',
                 gate_bias_initializer='zeros',
                 expert_bias_regularizer=None,
                 gate_bias_regularizer=None,
                 expert_bias_constraint=None,
                 gate_bias_constraint=None,
                 expert_kernel_initializer='VarianceScaling',
                 gate_kernel_initializer='VarianceScaling',
                 expert_kernel_regularizer=None,
                 gate_kernel_regularizer=None,
                 expert_kernel_constraint=None,
                 gate_kernel_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        """
        :param expert_units: Number of expert net hidden units
        :param num_experts: Number of experts
        :param num_tasks: Number of tasks
        :param gate_units: Number of gate net hidden units
        """
        super().__init__(**kwargs)

        # Hidden nodes parameter
        self.expert_units = expert_units
        self.num_experts = num_experts
        self.gate_units = gate_units
        self.num_tasks = num_tasks

        # Weight parameter
        self.expert_kernels = None
        self.gate_kernels = None
        self.expert_kernel_initializer = initializers.get(expert_kernel_initializer)
        self.gate_kernel_initializer = initializers.get(gate_kernel_initializer)
        self.expert_kernel_regularizer = regularizers.get(expert_kernel_regularizer)
        self.gate_kernel_regularizer = regularizers.get(gate_kernel_regularizer)
        self.expert_kernel_constraint = constraints.get(expert_kernel_constraint)
        self.gate_kernel_constraint = constraints.get(gate_kernel_constraint)

        # Activation parameter
        #self.expert_activation = activations.get(expert_activation)
        self.expert_activation = expert_activation
        self.gate_activation = gate_activation

        # Bias parameter
        self.expert_bias = None
        self.gate_bias = None
        self.use_expert_bias = use_expert_bias
        self.use_gate_bias = use_gate_bias
        self.expert_bias_initializer = initializers.get(expert_bias_initializer)
        self.gate_bias_initializer = initializers.get(gate_bias_initializer)
        self.expert_bias_regularizer = regularizers.get(expert_bias_regularizer)
        self.gate_bias_regularizer = regularizers.get(gate_bias_regularizer)
        self.expert_bias_constraint = constraints.get(expert_bias_constraint)
        self.gate_bias_constraint = constraints.get(gate_bias_constraint)

        # Activity parameter
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.expert_layers = []
        self.gate_layers = []
        for i in range(self.num_experts):
            self.expert_layers.append(MLP(out_dim=self.expert_units[-1],hidden_units=self.expert_units[:-1],
                                                   activation=self.expert_activation,
                                                   use_bias=self.use_expert_bias,
                                                   kernel_initializer=self.expert_kernel_initializer,
                                                   bias_initializer=self.expert_bias_initializer,
                                                   kernel_regularizer=self.expert_kernel_regularizer,
                                                   bias_regularizer=self.expert_bias_regularizer,
                                                   activity_regularizer=None,
                                                   kernel_constraint=self.expert_kernel_constraint,
                                                   bias_constraint=self.expert_bias_constraint))
        for i in range(self.num_tasks):
            self.gate_layers.append(MLP(out_dim=self.num_experts,hidden_units=self.gate_units,
                                                 activation=self.gate_activation,
                                                 use_bias=self.use_gate_bias,
                                                 kernel_initializer=self.gate_kernel_initializer,
                                                 bias_initializer=self.gate_bias_initializer,
                                                 kernel_regularizer=self.gate_kernel_regularizer,
                                                 bias_regularizer=self.gate_bias_regularizer, activity_regularizer=None,
                                                 kernel_constraint=self.gate_kernel_constraint,
                                                 bias_constraint=self.gate_bias_constraint))

    def call(self, inputs):
        """
        inputs: (bs,field_num*emb_size)
        ouput: list: num_task *(bs,expert_units[-1])
        """
        expert_outputs, gate_outputs, final_outputs = [], [], []
        for expert_layer in self.expert_layers:
            expert_output = expand_dims(expert_layer(inputs), axis=2) # [batch,expert_units[-1],1]
            expert_outputs.append(expert_output)
        expert_outputs = tf.concat(expert_outputs,2) # [batch,expert_units[-1],num_experts]
        
        for gate_layer in self.gate_layers:
            gate_outputs.append(gate_layer(inputs))

        for gate_output in gate_outputs:
            expanded_gate_output = expand_dims(gate_output, axis=1) # [batch,1,num_experts]
            ## repeat_elements -> [batch,expert_units[-1],num_experts]
            weighted_expert_output = expert_outputs * repeat_elements(expanded_gate_output, self.expert_units[-1], axis=1)
            ## sum [batch,expert_units[-1],num_experts]-> [batch,expert_units[-1]]
            final_outputs.append(sum(weighted_expert_output, axis=2))
        # 返回的矩阵维度 num_tasks * [ batch * expert_units[-1] ]
        return final_outputs

if __name__ == "__main__":
    model=MMoELayer(
            num_experts=4,
            expert_units=[256],
            num_tasks=2,
            gate_units=[128,128]
    )
    x=tf.ones((64,128))
    out1,out2=model(x)
