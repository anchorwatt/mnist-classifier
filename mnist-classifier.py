import math
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from mnist_helper import read, one_hot, flatten_X

print("importing data")
train_data = list(read(dataset="training", path="images/"))
test_data = list(read(dataset="testing", path="images/"))


print("arranging data")
Y_train = one_hot([pair[0] for pair in train_data[0:50000]])
Y_dev = one_hot([pair[0] for pair in train_data[50000:60000]])
Y_test = one_hot([pair[0] for pair in test_data])


X_train = flatten_X([pair[1] for pair in train_data[0:50000]])
X_dev = flatten_X([pair[1] for pair in train_data[50000:60000]])
X_test = flatten_X([pair[1] for pair in test_data])


def get_layer(X, num_units, activation=tf.nn.relu):
    # X: matrix containing training data of dimension [num_examples, input_length]
    # input_length: #columns in input matrix
    # num_units: #columns in output matrix
    # activation: function applied after matrix multiplication/addition
    # layer_num: number of layer in the network
    input_length = X.shape[1]
    W = tf.get_variable("W", [input_length, num_units], initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
    b = tf.get_variable("b", [num_units], initializer=tf.zeros_initializer())
    outputs = tf.add(tf.matmul(X, W), b)
    if activation is not None:
        outputs = activation(outputs)
    mean, var = tf.nn.moments(outputs, [1], keep_dims=True)
    outputs = tf.divide(tf.subtract(outputs, mean), tf.sqrt(var))
    return outputs, W, b


def build_network(X, nodes):
    # X: the input matrix
    # nodes: size of each hidden layer
    print("building network")
    with tf.variable_scope("layer1"):
        l1, W1, b1 = get_layer(X, nodes[0])
    with tf.variable_scope("layer2"):
        l2, W2, b2 = get_layer(l1, nodes[1])
    with tf.variable_scope("layer3"):
        l3, W3, b3 = get_layer(l2, nodes[2])
    with tf.variable_scope("layer4"):
        l4, W4, b4 = get_layer(l3, nodes[3], activation=None)
    weights = [W1, W2, W3, W4]
    return l4, weights


def l2norm(weights):
    # weights: weight layers of the neural network
    norm = 0
    for weight in weights:
        norm += tf.sqrt(tf.reduce_sum(tf.square(weight)))
    return norm


def get_cost(log, lab, weights, lambd=0):
    # log: output layer of the neural net
    # lab: example labels
    # lambda: regularization coefficient
    m = tf.shape(log)[0]
    cross_entropy_term = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=log, labels=lab)) 
    reg_coef = tf.cast(tf.divide(lambd, 2*m), tf.float32)
    norm = l2norm(weights)
    return cross_entropy_term + tf.multiply(reg_coef, norm)


def fit_model(X_train, Y_train, num_epochs=10, lambd=0, learning_rate=0.001, minibatch_size=256, dev=True, test=True):
    print("fitting model")

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    X = tf.placeholder(tf.float32, shape=[None, 784], name="X")
    Y = tf.placeholder(tf.float32, shape=[None, 10], name="Y")

    nodes = [500, 500, 200, 10]

    with tf.variable_scope("mnist"):
        pred, weights = build_network(X, nodes)
        tf.get_variable_scope().reuse_variables()
        cost = get_cost(pred, Y, weights, lambd=lambd)
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


fit_model(X_train, Y_train, num_epochs=20, lambd=1)