from __future__ import division, print_function, absolute_import

import tensorflow as tf
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data


class AutoEncoder(object):
    def __init__(self, layer1, layer2, data):
        self.num_hidden_1 = layer1  # 1st layer num features
        self.num_hidden_2 = layer2
        num_input = data.shape[1]
        self.X = tf.placeholder("float", [None, num_input])

        self.weights = {
            'encoder_h1': tf.Variable(tf.random_normal([num_input, self.num_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([self.num_hidden_1, self.num_hidden_2])),
            'decoder_h1': tf.Variable(tf.random_normal([self.num_hidden_2, self.num_hidden_1])),
            'decoder_h2': tf.Variable(tf.random_normal([self.num_hidden_1, num_input])),
        }
        self.biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([self.num_hidden_2])),
            'decoder_b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
            'decoder_b2': tf.Variable(tf.random_normal([num_input])),
        }
        self.encoder_op = self.encoder(self.X)
        self.decoder_op = self.decoder(self.encoder_op)
        self.y_pred = self.decoder_op
        self.y_true = self.X

        self.loss = tf.reduce_mean(tf.pow(self.y_true - self.y_pred, 2))

        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()


    def train(self, data, test_data, alpha=0.001, batch_size=20, num_steps=100):
        # num_input = 784
        # n_batches = int(data.shape[0] / batch_size)

        # tf.summary.scalar("Training/Testing loss", loss)

        optimizer = tf.train.RMSPropOptimizer(alpha).minimize(self.loss)

        # init = tf.global_variables_initializer()

        # mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

        # merged = tf.summary.merge_all()

        # with tf.Session() as sess:
        #     # summary_writer = tf.summary.FileWriter('./graph/', sess.graph)
        #     sess.run(init)

        self.sess.run(tf.global_variables_initializer())
        for i in range(1, num_steps + 1):
            # for j in range(n_batches):
            for j in range(100):
                # batch_x = mnist.train.next_batch(100)[0]
                # _, l, summary = sess.run([optimizer, loss, merged], feed_dict={X: batch_x})
                batch_x = data[j * batch_size:(j * batch_size + batch_size), :]
                _, l = self.sess.run([optimizer, self.loss], feed_dict={self.X: batch_x})

                # if i % 1 == 0:
                #     print('Step %i: Minibatch Loss: %f' % (i, l))

            test_error = self.predict(testdata=test_data)

            print('Predict Generalization Error at {0} : {1} '.format(i, str(test_error)))
                # summary_writer.add_summary(summary, i)

    def encoder(self, x):
        with tf.name_scope("Encoder"):
            layer_1 = tf.nn.relu(tf.add(tf.matmul(x, self.weights['encoder_h1']),
                                        self.biases['encoder_b1']))
            layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']),
                                        self.biases['encoder_b2']))
            return layer_2

    def decoder(self, x):
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, self.weights['decoder_h1']),
                                    self.biases['decoder_b1']))
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']),
                                    self.biases['decoder_b2']))
        return layer_2

    def predict(self, testdata):
        # num_input = testdata.shape[1]
        # # num_input = 784
        # testX = tf.placeholder("float", [None, num_input])

        # encoder_op = self.encoder(testX)
        # decoder_op = self.decoder(encoder_op)

        # y_pred = decoder_op
        # y_true = testX

        # reconstruction_error = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

        # tf.summary.scalar("Generalization Error", reconstruction_error)

        result = self.sess.run([self.loss], feed_dict={self.X: testdata})

        # assert (testdata.shape[0] == len(result[0]))

        return result[0]
