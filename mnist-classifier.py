import math
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from mnist_helper import read, one_hot, flatten_X


train_data = list(read(dataset="training", path="images/"))
test_data = list(read(dataset="testing", path="images/"))


Y_train = one_hot([pair[0] for pair in train_data[0:50000]])
Y_dev = one_hot([pair[0] for pair in train_data[50000:60000]])
Y_test = one_hot([pair[0] for pair in test_data])


X_train = flatten_X([pair[1] for pair in train_data[0:50000]])
X_dev = flatten_X([pair[1] for pair in train_data[50000:60000]])
X_test = flatten_X([pair[1] for pair in test_data])


def get_layer(X, input_length, num_units, activation=tf.nn.relu):
    # X: matrix containing training data of dimension [num_examples, input_length]
    # input_length: #columns in input matrix
    # num_units: #columns in output matrix
    # activation: function applied after matrix multiplication/addition
    # layer_num: number of layer in the network
    W = tf.get_variable("W", [input_length, num_units], initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
    b = tf.get_variable("b", [num_units], initializer=tf.zeros_initializer())
    outputs = tf.add(tf.matmul(X, W), b)
    if activation is not None:
        outputs = activation(outputs)
    mean, var = tf.nn.moments(outputs, [1], keep_dims=True)
    outputs = tf.divide(tf.subtract(outputs, mean), tf.sqrt(var))
    return outputs


def build_network(X):
    # X: the input matrix
    with tf.variable_scope("layer1"):
        l1 = get_layer(X, 784, 500)
    with tf.variable_scope("layer2"):
        l2 = get_layer(l1, 500, 500)
    with tf.variable_scope("layer3"):
        l3 = get_layer(l2, 500, 200)
    with tf.variable_scope("layer4"):
        l4 = get_layer(l3, 200, 10, activation=None)
    return l4


def get_cost(log, lab):
    # log: output layer of the neural net
    # lab: example labels
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=log, labels=lab)) 


def fit_model(X_train, Y_train, X_test, Y_test, minibatch_size=256, num_epochs=50, learning_rate=0.001, test=True):
    seed = 1

    X_train = np.array(X_train)
    X_train = ((X_train.transpose() - np.mean(X_train, axis=1)) / np.std(X_train, axis=1)).transpose() # normalize the data

    X = tf.placeholder(tf.float32, shape=[None, 784], name="X")
    Y = tf.placeholder(tf.float32, shape=[None, 10], name="Y")

    pred = build_network(X)
    cost = get_cost(pred, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) 

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        correct = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        correct_pct = 100*tf.reduce_mean(tf.cast(correct, "float"))

        for epoch in range(num_epochs):
            epoch_cost = 0

            num_minibatches = math.ceil(len(X_train)/minibatch_size)
            idx = np.random.choice(len(X_train), len(X_train)) # generate permutation of indices

            for iteration in range(num_minibatches):
                start_idx = minibatch_size * iteration
                X_mini = [X_train[idx[i]] for i in range(start_idx, min(start_idx + minibatch_size, len(idx)))]
                Y_mini = [Y_train[idx[i]] for i in range(start_idx, min(start_idx + minibatch_size, len(idx)))]
                _ , minibatch_cost, out = sess.run([optimizer, cost, pred], feed_dict={X:X_mini, Y:Y_mini})
                epoch_cost += minibatch_cost

            epoch_cost /= num_minibatches
            _, acc = sess.run([optimizer, correct_pct], feed_dict={X:X_train, Y:Y_train})

            print("epoch {}: cost={}, accuracy={}%".format(epoch, epoch_cost, acc))

        print("final cost={}".format(cost))
        print("final accuracy: {}%".format(acc))


fit_model(X_train, Y_train, X_test, Y_test)

