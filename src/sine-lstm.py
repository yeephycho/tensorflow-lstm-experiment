import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import os
import time

Fs = 1000
f = 0.1
sample = 100000
x = np.arange(sample)
y = np.reshape(np.sin(2 * np.pi * f * x / Fs), (-1, 100))
print(y.shape)

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev = 0.01))

W = init_weights([100, 100])
B = init_weights([100])


n_hidden_1 = 5 # 1st layer number of features
n_hidden_2 = 5 # 2nd layer number of features
n_input = 100 # lstm output
n_classes = 100 # final output

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


def model(x):
    # Make lstm with lstm_size (each input vector size)
#    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)

    X = tf.reshape(x, [-1, 100])

    X_split = tf.split(X, 10, 1)

    lstm = tf.contrib.rnn.BasicLSTMCell(10)
    # Get lstm cell output, time_step_size (28) arrays with lstm_size output: (batch_size, lstm_size)
    outputs, _states = tf.contrib.rnn.static_rnn(lstm, X_split, dtype=tf.float32)
    print(outputs)
    # Linear activation
    # Get the last output
    return multilayer_perceptron(tf.reshape(outputs, (-1, 100)), weights, biases), lstm.state_size # State size to initialize the stat

X = tf.placeholder("float", [None, 100])
Y = tf.placeholder("float", [None, 100])

py_x, state_size = model(X)

#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = py_x))
cost = tf.reduce_mean(tf.losses.mean_squared_error(labels = Y, predictions = py_x))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = py_x

saver = tf.train.Saver()
inf_out = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "../model/sine-lstm.ckpt")
    for j in range(30):
        print("Epoch %d" % j)
        for i in range(900):
            print("shape of the y is: {0}".format(np.reshape(y[i, :], [-1, 100]).shape))
            _, loss = sess.run([train_op, cost], feed_dict={X: np.reshape(y[i, :], [-1, 100]), Y: (np.reshape(y[i+1, :], [-1, 100]))})
            print(loss)

#        test_indices = np.arange(len(x))
#        np.random.shuffle(test_indices)
#        test_indices = test_indices[0:test_size]

#        print(i, np.mean(np.argmax(teY[test_indices], axis = 1) == sess.run(predict_op, feed_dict = {X: teX[test_indices], Y: teY[test_indices]})))
    saver.save(sess, "../model/sine-lstm.ckpt")


    for i in range(900):
        prediction = sess.run(predict_op, feed_dict={X: np.reshape(y[i, :], [-1, 100])})
        inf_out.append(prediction)
    sess.close()



plt.plot(np.asarray(inf_out).flatten())
plt.show()

plt.plot(np.asarray(y.flatten()))
plt.show()
