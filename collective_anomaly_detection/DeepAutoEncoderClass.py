from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import pandas

from collective_anomaly_detection.data_processing import normalize_data

class AutoEncoder(object):

    def __init__(self):
        self.num_hidden_1 = 10  # 1st layer num features
        self.num_hidden_2 = 5

    def train(self, data, alpha=0.001, batch_size=20, num_steps=100):
        num_input = data.shape[1]
        n_batches = int(data.shape[0] / batch_size)
        X = tf.placeholder("float", [None, num_input])

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

        encoder_op = self.encoder(X)
        decoder_op = self.decoder(encoder_op)
        y_pred = decoder_op
        y_true = X

        loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

        optimizer = tf.train.AdamOptimizer(alpha).minimize(loss)


        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            for i in range(1, num_steps + 1):
                for j in range(n_batches):
                    batch_x = data[j * batch_size:(j * batch_size + batch_size), ]

                    _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})

                if i % 1 == 0:
                    print('Step %i: Minibatch Loss: %f' % (i, l))
                    # summary_writer.add_summary(summary, i)

    def encoder(self, x):
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
        num_input = testdata.shape[1]
        testX = tf.placeholder("float", [None, num_input])

        encoder_op = self.encoder(testX)
        decoder_op = self.decoder(encoder_op)

        y_pred = decoder_op
        y_true = testX

        reconstruction_error = tf.pow(y_true - y_pred, 2)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            result = sess.run([reconstruction_error], feed_dict={testX: testdata})

        assert(testdata.shape[0] == len(result[0]))

        return result[0]