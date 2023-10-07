#coding:utf-8
import tensorflow as tf
from tensorflow.keras import layers,initializers, regularizers, activations, constraints
from tensorflow.keras.backend import expand_dims,repeat_elements,sum

from ..base import MLP

class PLELayer(layers.Layer):

    def __init__(self,
                 num_per_experts,
                 num_experts_share,
                 expert_units,
                 num_tasks,
                 gate_units,
                 level_number,
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
        :param level_number: Number of PLELayer
        """
        super(PLELayer, self).__init__(**kwargs)

        # Hidden nodes parameter
        self.expert_units = expert_units
        self.num_per_experts = num_per_experts
        self.num_experts_share = num_experts_share
        self.num_tasks = num_tasks
        self.gate_units=gate_units
        self.level_number = level_number

        # ple layer
        self.ple_layers = []
        for i in range(0, self.level_number):
            if i == self.level_number - 1:
                ple_layer = SinglePLELayer(
                    num_per_experts, num_experts_share,expert_units, num_tasks,gate_units, True)
                self.ple_layers.append(ple_layer)
                break
            else:
                ple_layer = SinglePLELayer(
                    num_per_experts, num_experts_share,expert_units, num_tasks,gate_units, False)
                self.ple_layers.append(ple_layer)
    
    def call(self, inputs):
        inputs_ple = []
        # task_num part + shared part
        for i in range(0, self.num_tasks + 1):
            inputs_ple.append(inputs)
        # multiple ple layer
        ple_out = []
        for i in range(0, self.level_number):
            ple_out = self.ple_layers[i](inputs_ple)
            inputs_ple = ple_out

        return ple_out

class SinglePLELayer(layers.Layer):
    def __init__(self,
                 num_per_experts,
                 num_experts_share,
                 expert_units,
                 num_tasks,
                 gate_units,
                 if_last,
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
        :param if_last: is or not last of SinglePLELayer
        """
        super(SinglePLELayer, self).__init__(**kwargs)

        # Hidden nodes parameter
        self.expert_units = expert_units
        self.gate_units=gate_units
        self.num_per_experts = num_per_experts
        self.num_experts_share = num_experts_share
        self.num_tasks = num_tasks
        self.if_last = if_last

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
        self.expert_share_layers = []
        self.gate_layers = []
        self.gate_share_layers = []
        # task-specific expert part
        for i in range(0, self.num_tasks):
            for j in range(self.num_per_experts):
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
        # task-specific expert part
        for i in range(self.num_experts_share):
            self.expert_share_layers.append(MLP(out_dim=self.expert_units[-1],hidden_units=self.expert_units[:-1],
                                                   activation=self.expert_activation,
                                                   use_bias=self.use_expert_bias,
                                                   kernel_initializer=self.expert_kernel_initializer,
                                                   bias_initializer=self.expert_bias_initializer,
                                                   kernel_regularizer=self.expert_kernel_regularizer,
                                                   bias_regularizer=self.expert_bias_regularizer,
                                                   activity_regularizer=None,
                                                   kernel_constraint=self.expert_kernel_constraint,
                                                   bias_constraint=self.expert_bias_constraint))
        # task gate part
        for i in range(self.num_tasks):
            self.gate_layers.append(MLP(out_dim=self.num_per_experts+self.num_experts_share,hidden_units=self.gate_units, 
                                                 activation=self.gate_activation,
                                                 use_bias=self.use_gate_bias,
                                                 kernel_initializer=self.gate_kernel_initializer,
                                                 bias_initializer=self.gate_bias_initializer,
                                                 kernel_regularizer=self.gate_kernel_regularizer,
                                                 bias_regularizer=self.gate_bias_regularizer, activity_regularizer=None,
                                                 kernel_constraint=self.gate_kernel_constraint,
                                                 bias_constraint=self.gate_bias_constraint))
        # task gate part
        if not if_last:
            self.gate_share_layers.append(MLP(out_dim=self.num_tasks*self.num_per_experts+self.num_experts_share,hidden_units=self.gate_units, 
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
        ouput: list: num_task *(bs,field_num*emb_size)
        """
        #assert input_shape is not None and len(input_shape) >= 2

        expert_outputs, expert_share_outputs, gate_outputs, final_outputs = [], [], [], []
        for i in range(0, self.num_tasks):
            for j in range(self.num_per_experts):
                expert_output = expand_dims(self.expert_layers[i*self.num_per_experts+j](inputs[i]), axis=2)
                expert_outputs.append(expert_output)
        for i in range(0, self.num_experts_share):
            expert_output = expand_dims(self.expert_share_layers[i](inputs[-1]), axis=2)
            expert_share_outputs.append(expert_output)

        for i in range(0, self.num_tasks):
            gate_outputs.append(self.gate_layers[i](inputs[i]))

        for i in range(0, self.num_tasks):
            expanded_gate_output = expand_dims(gate_outputs[i], axis=1)
            cur_expert = []
            cur_expert = cur_expert + expert_outputs[i*self.num_per_experts:(i+1)*self.num_per_experts]
            cur_expert = cur_expert + expert_share_outputs
            cur_expert = tf.concat(cur_expert,2)
            weighted_expert_output = cur_expert * repeat_elements(expanded_gate_output, self.expert_units[-1], axis=1)
            final_outputs.append(sum(weighted_expert_output, axis=2))

        if not self.if_last:
            gate_share_outputs = []
            for gate_layer in self.gate_share_layers:
                gate_share_outputs.append(gate_layer(inputs[-1]))
            expanded_gate_output = expand_dims(gate_share_outputs[0], axis=1)
            cur_expert = []
            cur_expert = cur_expert + expert_outputs
            cur_expert = cur_expert + expert_share_outputs
            cur_expert = tf.concat(cur_expert,2)
            weighted_expert_output = cur_expert * repeat_elements(expanded_gate_output, self.expert_units[-1], axis=1)
            final_outputs.append(sum(weighted_expert_output, axis=2))
        # 返回的矩阵维度 num_tasks * batch * units
        return final_outputs

