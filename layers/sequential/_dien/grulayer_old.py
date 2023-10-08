# https://blog.csdn.net/qq_42363032/article/details/122365548
import tensorflow as tf

from tensorflow.keras import layers

'''   AGRU单元   '''
class AGRUCell(layers.Layer):
    """
        Attention based GRU (AGRU)
        公式如下:
            r = sigmoid(W_ir * x + b_ir + W_hr * h + b_hr)
            #z = sigmoid(W_iz * x + b_iz + W_hz * h + b_hz)
            h' = tanh(W_ih * x + b_ih + r * (W_hh * h + b_hh))
            h = (1 - att_score) * h + att_score * h'

    """
    def __init__(self, units):
        super().__init__()
        self.units = units
        # 作为一个 RNN 的单元，必须有state_size属性
        # state_size 表示每个时间步输出的维度
        self.state_size = units

    def build(self, input_shape):
        # 输入数据是一个tupe: (gru_embed, atten_scores)  (2000, 4)、(2000, 1)
        # 因此，t时刻输入的维度为：
        dim_xt = input_shape[0][-1]

        # 重置门中的参数
        self.w_ir = tf.Variable(tf.random.normal(shape=[dim_xt, self.units]), name='w_ir')
        self.w_hr = tf.Variable(tf.random.normal(shape=[self.units, self.units]), name='w_hr')
        self.b_ir = tf.Variable(tf.random.normal(shape=[self.units]), name='b_ir')
        self.b_hr = tf.Variable(tf.random.normal(shape=[self.units]), name='b_hr')
        # 更新门被att_score代替

        # 候选隐藏中的参数
        self.w_ih = tf.Variable(tf.random.normal(shape=[dim_xt, self.units]), name='w_ih')
        self.w_hh = tf.Variable(tf.random.normal(shape=[self.units, self.units]), name='w_hh')
        self.b_ih = tf.Variable(tf.random.normal(shape=[self.units]), name='b_ih')
        self.b_hh = tf.Variable(tf.random.normal(shape=[self.units]), name='b_hh')

    def call(self, inputs, states):
        x_t, att_score = inputs
        states = states[0]

        # 重置门
        r_t = tf.sigmoid(tf.matmul(x_t, shttps://blog.csdn.net/qq_42363032/article/details/122365548elf.w_ir) + self.b_ir + \
                         tf.matmul(states, self.w_hr) + self.b_hr)

        # 候选隐藏状态
        h_t_ = tf.tanh(tf.matmul(x_t, self.w_ih) + self.b_ih + \
                       tf.multiply(r_t, (tf.matmul(states, self.w_hh) + self.b_hh)))
        # 输出值
        h_t = tf.multiply(1 - att_score, states) + tf.multiply(att_score, h_t_)
        # 对gru而言，当前时刻的output与传递给下一时刻的state相同
        next_state = h_t

        return h_t, next_state

'''   AUGRU单元   '''
class AUGRUCell(layers.Layer):
    """
        GRU with attentional update gate (AUGRU)
        公式如下:
            r = sigmoid(W_ir * x + b_ir + W_hr * h + b_hr)
            z = sigmoid(W_iz * x + b_iz + W_hz * h + b_hz)
            z = z * att_score
            h' = tanh(W_ih * x + b_ih + r * (W_hh * h + b_hh))
            h = (1 - z) * h + z * h'

    """
    def __init__(self, units):
        super().__init__()
        self.units = units
        # 作为一个 RNN 的单元，必须有state_size属性
        # state_size 表示每个时间步输出的维度
        self.state_size = units

    def build(self, input_shape):
        # 输入数据是一个tupe: (gru_embed, atten_scores)
        # 因此，t时刻输入的维度为：
        dim_xt = input_shape[0][-1]

        # 重置门中的参数
        self.w_ir = tf.Variable(tf.random.normal(shape=[dim_xt, self.units]), name='w_ir')
        self.w_hr = tf.Variable(tf.random.normal(shape=[self.units, self.units]), name='w_hr')
        self.b_ir = tf.Variable(tf.random.normal(shape=[self.units]), name='b_ir')
        self.b_hr = tf.Variable(tf.random.normal(shape=[self.units]), name='b_hr')

        # 更新门中的参数
        self.w_iz = tf.Variable(tf.random.normal(shape=[dim_xt, self.units]), name='w_iz')
        self.w_hz = tf.Variable(tf.random.normal(shape=[self.units, self.units]), name='W_hz')
        self.b_iz = tf.Variable(tf.random.normal(shape=[self.units]), name='b_iz')
        self.b_hz = tf.Variable(tf.random.normal(shape=[self.units]), name='b_hz')

        # 候选隐藏中的参数
        self.w_ih = tf.Variable(tf.random.normal(shape=[dim_xt, self.units]), name='w_ih')
        self.w_hh = tf.Variable(tf.random.normal(shape=[self.units, self.units]), name='w_hh')
        self.b_ih = tf.Variable(tf.random.normal(shape=[self.units]), name='b_ih')
        self.b_hh = tf.Variable(tf.random.normal(shape=[self.units]), name='b_hh')

    def call(self, inputs, states):
        x_t, att_score = inputs
        states = states[0]
        # 重置门
        r_t = tf.sigmoid(tf.matmul(x_t, self.w_ir) + self.b_ir + \
                         tf.matmul(states, self.w_hr) + self.b_hr)
        # 更新门
        z_t = tf.sigmoid(tf.matmul(x_t, self.w_iz) + self.b_iz + \
                         tf.matmul(states, self.w_hz) + self.b_hz)
        # 带有注意力的更新门
        z_t = tf.multiply(att_score, z_t)
        # 候选隐藏状态
        h_t_ = tf.tanh(tf.matmul(x_t, self.w_ih) + self.b_ih + \
                       tf.multiply(r_t, (tf.matmul(states, self.w_hh) + self.b_hh)))
        # 输出值
        h_t = tf.multiply(1 - z_t, states) + tf.multiply(z_t, h_t_)
        # 对gru而言，当前时刻的output与传递给下一时刻的state相同
        next_state = h_t

        return h_t, next_state
