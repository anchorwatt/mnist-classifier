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


def get_layer(X, input_length, num_units, activation=tf.nn.relu, scope=None):
    # X: matrix containing training data of dimension [num_examples, input_length]
    # input_length: #columns in input matrix
    # num_units: #columns in output matrix
    # activation: function applied after matrix multiplication/addition
    # scope: local scope for variable names
    with tf.variable_scope(scope):
        W = tf.get_variable("W", [input_length, num_units], initializer=tf.random_normal_initializer())
        b = tf.get_variable("b", [num_units], initializer=tf.zeros_initializer())
        outputs = tf.add(tf.matmul(X, W), b)
        if activation is not None:
            outputs = activation(outputs)
        return outputs


def get_cost(log, lab):
    # log: output layer of the neural net
    # lab: example labels
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=log, labels=lab))   
    return cost


def fit_model(X_in, Y_in, num_epochs=500, minibatch_size=32, learning_rate=0.0001):
    tf.reset_default_graph()
    seed = 1

    X = tf.placeholder(tf.float32, shape=[None, 784], name="X")
    layer1 = get_layer(X, 784, 500, scope="l1")
    layer2 = get_layer(layer1, 500, 200, scope="l2")
    layer3 = get_layer(layer2, 200, 100, scope="l3")
    output = get_layer(layer3, 100, 10, scope="out")

    Y = tf.placeholder(tf.float32, shape=[None, 10], name="Y")
    cost = get_cost(output, Y)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost) 
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = int(len(X_in)/minibatch_size)
            seed = seed + 1

            for iteration in range(num_minibatches):
                idx = np.random.choice(50000, minibatch_size)
                X_mini = [X_in[i] for i in idx]
                Y_mini = [Y_in[i] for i in idx]
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X:X_mini, Y:Y_mini}) 
                epoch_cost += minibatch_cost/num_minibatches

            print("epoch {}: cost={}".format(epoch, epoch_cost))

        pred = sess.run(output, feed_dict={X:X_in, Y:Y_in})
        correct = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        correct_pct = tf.reduce_mean(tf.cast(correct, "float"))

        acc = sess.run([correct_pct], feed_dict={X:X_in, Y:Y_in})
        print("accuracy: {}%".format(acc*100))
            


fit_model(X_train, Y_train)

