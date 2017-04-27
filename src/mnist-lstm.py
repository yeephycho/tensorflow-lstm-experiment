import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import os
import time

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot = True)
print("Number of training data is %d" % (mnist.train.num_examples))
print("Number of test data is %d" % (mnist.test.num_examples))

nsample = 3

rand_idx = np.random.randint(mnist.train.images.shape[0], size = nsample)
print(rand_idx)

for i in rand_idx:
    curr_img = np.reshape(mnist.train.images[i, :], (28, 28))
    curr_lbl = np.argmax(mnist.train.labels[i, :])
    plt.matshow(curr_img, cmap = plt.get_cmap("gray"))
    plt.title("" + str(i) + "th training image " + "(label: " + str(curr_lbl) + ")")
    plt.show()


input_vec_size = 28
lstm_vec_num = 28
lstm_size = 28
time_step_size = 28

batch_size = 128
test_size = 256

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev = 0.01))


def model(X, W, B, lstm_size):
    # X, input shape: (batch_size, time_step_size, input_vec_size)
    XT = tf.transpose(X, [1, 0, 2])  # permute time_step_size and batch_size
    # XT shape: (time_step_size, batch_size, input_vec_size)
    XR = tf.reshape(XT, [-1, lstm_vec_num]) # each row has input for each lstm cell (lstm_size=input_vec_size)
    # XR shape: (time_step_size * batch_size, input_vec_size)
    X_split = tf.split(XR, time_step_size, 0) # split them to time_step_size (28 arrays)
    # Each array shape: (batch_size, input_vec_size)

    # Make lstm with lstm_size (each input vector size)
#    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    # Get lstm cell output, time_step_size (28) arrays with lstm_size output: (batch_size, lstm_size)
    outputs, _states = tf.contrib.rnn.static_rnn(lstm, X_split, dtype=tf.float32)
#    print(outputs)
#    tensor_outputs = outputs[len(outputs) - 28 : len(outputs)]
#    print(tensor_outputs)
    # Linear activation
    # Get the last output
    return tf.matmul(outputs[-1], W) + B, lstm.state_size # State size to initialize the stat


trX = mnist.train.images
trY = mnist.train.labels
teX = mnist.test.images
teY = mnist.test.labels

trX = trX.reshape(-1, 28, 28)
teX = teX.reshape(-1, 28, 28)

X = tf.placeholder("float", [None, 28, 28])
Y = tf.placeholder("float", [None, 10])

W = init_weights([lstm_size, 10])
B = init_weights([10])

py_x, state_size = model(X, W, B, lstm_size)
print(state_size)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels = Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "../model/mnist-lstm.ckpt")
    for i in range(10):
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX) + 1, batch_size)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

        test_indices = np.arange(len(teX))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(teY[test_indices], axis = 1) == sess.run(predict_op, feed_dict = {X: teX[test_indices], Y: teY[test_indices]})))
    saver.save(sess, "../model/mnist-lstm.ckpt")
    sess.close()
