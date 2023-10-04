#coding:utf-8
import tensorflow as tf
from tensorflow import tensordot, expand_dims
from tensorflow.keras import layers, Model, initializers, regularizers, activations, constraints, Input
from layers.base import MLP

from tensorflow.keras.backend import expand_dims,repeat_elements,sum

class PLELayer(layers.Layer):

    def __init__(self,
                 num_per_experts,
                 num_experts_share,
                 expert_units,
                 num_tasks,
                 gate_units,
                 level_number,
                 expert_activation='relu',
                 gate_activation='softmax',
                 **kwargs):
        """
         Method for instantiating MMoE layer.
        :param units: Number of hidden units
        :param num_per_experts: Number of experts for per task
        :param num_experts_share: Number of share experts
        :param num_tasks: Number of tasks
        :param level_number: Number of PLELayer
        :param use_expert_bias: Boolean to indicate the usage of bias in the expert weights
        :param use_gate_bias: Boolean to indicate the usage of bias in the gate weights
        :param expert_activation: Activation function of the expert weights
        :param gate_activation: Activation function of the gate weights
        :param expert_bias_initializer: Initializer for the expert bias
        :param gate_bias_initializer: Initializer for the gate bias
        :param expert_bias_regularizer: Regularizer for the expert bias
        :param gate_bias_regularizer: Regularizer for the gate bias
        :param expert_bias_constraint: Constraint for the expert bias
        :param gate_bias_constraint: Constraint for the gate bias
        :param expert_kernel_initializer: Initializer for the expert weights
        :param gate_kernel_initializer: Initializer for the gate weights
        :param expert_kernel_regularizer: Regularizer for the expert weights
        :param gate_kernel_regularizer: Regularizer for the gate weights
        :param expert_kernel_constraint: Constraint for the expert weights
        :param gate_kernel_constraint: Constraint for the gate weights
        :param activity_regularizer: Regularizer for the activity
        :param kwargs: Additional keyword arguments for the Layer class
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
    """
    SinglePLELayer Multi-gate Mixture-of-Experts model.
    """

    def __init__(self,
                 num_per_experts,
                 num_experts_share,
                 expert_units,
                 num_tasks,
                 gate_units,
                 if_last,
                 expert_activation='relu',
                 gate_activation='softmax',
                 **kwargs):
        """
         Method for instantiating MMoE layer.
        :param units: Number of hidden units
        :param num_per_experts: Number of experts for per task
        :param num_experts_share: Number of share experts
        :param num_tasks: Number of tasks
        :param if_last: is or not last of SinglePLELayer
        :param use_expert_bias: Boolean to indicate the usage of bias in the expert weights
        :param use_gate_bias: Boolean to indicate the usage of bias in the gate weights
        :param expert_activation: Activation function of the expert weights
        :param gate_activation: Activation function of the gate weights
        :param expert_bias_initializer: Initializer for the expert bias
        :param gate_bias_initializer: Initializer for the gate bias
        :param expert_bias_regularizer: Regularizer for the expert bias
        :param gate_bias_regularizer: Regularizer for the gate bias
        :param expert_bias_constraint: Constraint for the expert bias
        :param gate_bias_constraint: Constraint for the gate bias
        :param expert_kernel_initializer: Initializer for the expert weights
        :param gate_kernel_initializer: Initializer for the gate weights
        :param expert_kernel_regularizer: Regularizer for the expert weights
        :param gate_kernel_regularizer: Regularizer for the gate weights
        :param expert_kernel_constraint: Constraint for the expert weights
        :param gate_kernel_constraint: Constraint for the gate weights
        :param activity_regularizer: Regularizer for the activity
        :param kwargs: Additional keyword arguments for the Layer class
        """
        super(SinglePLELayer, self).__init__(**kwargs)

        # Hidden nodes parameter
        self.expert_units = expert_units
        self.gate_units=gate_units
        self.num_per_experts = num_per_experts
        self.num_experts_share = num_experts_share
        self.num_tasks = num_tasks
        self.if_last = if_last

        # Activation parameter
        #self.expert_activation = activations.get(expert_activation)
        self.expert_activation = expert_activation
        self.gate_activation = gate_activation

        # Activity parameter
        self.expert_layers = []
        self.expert_share_layers = []
        self.gate_layers = []
        self.gate_share_layers = []
        # task-specific expert part
        for i in range(0, self.num_tasks):
            for j in range(self.num_per_experts):
                self.expert_layers.append(MLP(self.expert_units[-1],self.expert_units[:-1],activation=self.expert_activation))
        # task-specific expert part
        for i in range(self.num_experts_share):
            self.expert_share_layers.append(MLP(self.expert_units[-1],self.expert_units[:-1],activation=self.expert_activation))
        # task gate part
        for i in range(self.num_tasks):
            self.gate_layers.append(MLP(self.num_per_experts+self.num_experts_share,self.gate_units,activation=self.gate_activation))
        # task gate part
        if not if_last:
            self.gate_share_layers.append(MLP(self.num_tasks*self.num_per_experts+self.num_experts_share,self.gate_units,activation=self.gate_activation))

    def call(self, inputs):
        """
        Method for the forward function of the layer.
        :param inputs: Input tensor
        :param kwargs: Additional keyword arguments for the base method
        :return: A tensor
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

