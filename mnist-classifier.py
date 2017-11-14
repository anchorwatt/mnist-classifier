import math
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from mnist_helper import *

print("importing data")
train_data = list(read(dataset="training", path="images/"))
test_data = list(read(dataset="testing", path="images/"))


print("arranging data")
Y_train = one_hot([pair[0] for pair in train_data[0:50000]])
Y_dev = one_hot([pair[0] for pair in train_data[50000:60000]])
Y_test = one_hot([pair[0] for pair in test_data])

X_train = expand_X([pair[1] for pair in train_data[0:50000]])
X_dev = expand_X([pair[1] for pair in train_data[50000:60000]])
X_test = expand_X([pair[1] for pair in test_data])


def build_network(X, lambd=0.1):
    # X: the input matrix
    print("building network")

    regularizer = tf.contrib.layers.l2_regularizer(scale=lambd)

    with tf.variable_scope("conv1"):
        Z1 = tf.layers.conv2d(X, filters=16, kernel_size=4, strides=1, padding="SAME")
        A1 = tf.nn.relu(Z1)
    with tf.variable_scope("pool1"):
        P1 = tf.layers.max_pooling2d(A1, pool_size=4, strides=4, padding="SAME")

    with tf.variable_scope("conv2"):
        Z2 = tf.layers.conv2d(P1, filters=32, kernel_size=4, strides=1, padding="SAME")
        A2 = tf.nn.relu(Z2)
    with tf.variable_scope("pool2"):
        P2 = tf.layers.max_pooling2d(A2, pool_size=4, strides=4, padding="SAME")

    """
    with tf.variable_scope("conv3"):
        Z3 = tf.layers.conv2d(P2, filters=32, kernel_size=5, strides=1, padding="SAME")
        A3 = tf.nn.relu(Z3)
    with tf.variable_scope("pool3"):
        P3 = tf.layers.max_pooling2d(A3, pool_size=4, strides=4, padding="SAME")
    """
    
    P3 = tf.contrib.layers.flatten(P2)
    Z4 = tf.contrib.layers.fully_connected(P3, 500, activation_fn=tf.nn.relu, weights_regularizer=regularizer)
    A4 = tf.nn.relu(Z4)

    Z5 = tf.contrib.layers.fully_connected(A4, 200, activation_fn=tf.nn.relu, weights_regularizer=regularizer)
    A5 = tf.nn.relu(Z5)

    """
    Z6 = tf.contrib.layers.fully_connected(A5, 100, activation_fn=tf.nn.relu, weights_regularizer=regularizer)
    A6 = tf.nn.relu(Z6)
    """

    pred = tf.contrib.layers.fully_connected(A5, 10, activation_fn=None, weights_regularizer=None)
    return pred


def fit_model(X_train, Y_train, num_epochs=10, lambd=0, learning_rate=0.005, minibatch_size=256, dev=True, test=True):
    print("fitting model")

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="X")
    Y = tf.placeholder(tf.float32, shape=[None, 10], name="Y")

    with tf.variable_scope("mnist"):
        pred = build_network(X, lambd=lambd)
        tf.get_variable_scope().reuse_variables()
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) 

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        correct = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        correct_pct = 100*tf.reduce_mean(tf.cast(correct, "float"))
        print("")

        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = math.ceil(len(X_train)/minibatch_size)

            idx = np.random.choice(len(X_train), len(X_train), replace=False) # generate permutation of indices
            X_train = X_train[idx]
            Y_train = Y_train[idx]

            for iteration in range(num_minibatches):
                start_idx = minibatch_size * iteration
                X_mini = [X_train[i] for i in range(start_idx, min(start_idx + minibatch_size, len(idx)))]
                Y_mini = [Y_train[i] for i in range(start_idx, min(start_idx + minibatch_size, len(idx)))]
                _ , minibatch_cost, out = sess.run([optimizer, cost, pred], feed_dict={X:X_mini, Y:Y_mini})
                epoch_cost += minibatch_cost

            epoch_cost /= num_minibatches
            _, train_acc = sess.run([optimizer, correct_pct], feed_dict={X:X_train, Y:Y_train})

            print("epoch {}".format(epoch))
            print("cost: {0:.5f}".format(epoch_cost))
            print("training accuracy: {0:.2f}%".format(train_acc))

            if(dev):
                _, dev_acc = sess.run([optimizer, correct_pct], feed_dict={X:X_test, Y:Y_test})
                print("development accuracy: {0:.2f}%".format(dev_acc))

            if(test):
                _, test_acc = sess.run([optimizer, correct_pct], feed_dict={X:X_test, Y:Y_test})
                print("testing accuracy: {0:.2f}%".format(test_acc))

            print("")


fit_model(X_train, Y_train, num_epochs=12, lambd=0.0)