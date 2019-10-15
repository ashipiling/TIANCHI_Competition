#更改tensorflow的版本
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

#添加层:输入数据，输入属性个数，输出属性个数，激励函数
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
#矩阵操作注意数据格式，统一为float32
    Wx_plus_b = tf.matmul(tf.cast(inputs,tf.float32), Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#转为300行1列的二维矩阵 x_data = x_data.reshape((300,1))
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

x = tf.placeholder(dtype = tf.float32, shape = x_data.shape)
y = tf.placeholder(dtype = tf.float32, shape = y_data.shape)

layer1 = add_layer(x_data, 1, 10, tf.nn.relu)
prediction = add_layer(layer1, 10, 1, None)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_data - prediction),reduction_indices = [1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(2000):
    sess.run(train_step, feed_dict = {x : x_data, y: y_data})
    if i % 100 ==  0:
        print(sess.run(loss))

sess.close()
