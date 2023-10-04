#coding:utf-8
import tensorflow as tf
from tensorflow import tensordot, expand_dims
from tensorflow.keras import layers, Model, initializers, regularizers, activations, constraints, Input


from tensorflow.keras.backend import expand_dims,repeat_elements,sum
from layers.base import MLP

class MMoE(layers.Layer):
    """
    Multi-gate Mixture-of-Experts model.
    expert_units 专家网络MLP结构expert_units[-1]为output
    gate_units ， gate_units是隐藏层，num_expert是output
    有多少 task 就有几个 gate
    """

    def __init__(self,
                 num_experts,
                 expert_units,
                 num_tasks,
                 gate_units,
                 expert_activation='relu',
                 gate_activation='softmax',
                 **kwargs):
        """
         Method for instantiating MMoE layer.
        :param units: Number of hidden units
        :param num_experts: Number of experts
        :param num_tasks: Number of tasks
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
        super(MMoE, self).__init__(**kwargs)

        # Hidden nodes parameter
        self.expert_units = expert_units
        self.num_experts = num_experts
        self.gate_units = gate_units
        self.num_tasks = num_tasks

        # Activation parameter
        #self.expert_activation = activations.get(expert_activation)
        self.expert_activation = expert_activation
        self.gate_activation = gate_activation

        self.expert_layers = []
        self.gate_layers = []
        for i in range(self.num_experts):
            self.expert_layers.append(MLP(self.expert_units[-1],self.expert_units[:-1],activation=self.expert_activation))
        for i in range(self.num_tasks):
            self.gate_layers.append(MLP(self.num_experts,self.gate_units, activation=self.gate_activation))

    def call(self, inputs):
        """
        Method for the forward function of the layer.
        :param inputs: Input tensor
        :param kwargs: Additional keyword arguments for the base method
        :return: A tensor
        """
        #assert input_shape is not None and len(input_shape) >= 2

        expert_outputs, gate_outputs, final_outputs = [], [], []
        for expert_layer in self.expert_layers:
            expert_output = expand_dims(expert_layer(inputs), axis=2)
            expert_outputs.append(expert_output)
        expert_outputs = tf.concat(expert_outputs,2)
        
        for gate_layer in self.gate_layers:
            gate_outputs.append(gate_layer(inputs))

        for gate_output in gate_outputs:
            expanded_gate_output = expand_dims(gate_output, axis=1)
            weighted_expert_output = expert_outputs * repeat_elements(expanded_gate_output, self.expert_units[-1], axis=1)
            final_outputs.append(sum(weighted_expert_output, axis=2))
        # 返回的矩阵维度 num_tasks * [ batch * expert_units[-1] ]
        return final_outputs
