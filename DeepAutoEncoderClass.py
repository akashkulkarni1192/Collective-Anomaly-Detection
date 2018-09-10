from __future__ import division, print_function, absolute_import

import tensorflow as tf



class AutoEncoder(object):

    def __init__(self, layer1, layer2, data, cache=False):
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
        self.recon_error = tf.pow(self.y_true - self.y_pred, 2)
        self.error = self.y_true - self.y_pred
        self.loss = tf.reduce_mean(self.recon_error)

        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        if cache:
            saver = tf.train.Saver()
            saver.restore(self.sess, "./models/autoencoder_processed.ckpt")


    def train(self, data, test_data, alpha=0.001, batch_size=20, num_steps=100):

        n_batches = int(data.shape[0] / batch_size)
        optimizer = tf.train.RMSPropOptimizer(alpha).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())
        for i in range(1, num_steps + 1):
            for j in range(n_batches):

                batch_x = data[j * batch_size:(j * batch_size + batch_size), :]
                _, l = self.sess.run([optimizer, self.loss], feed_dict={self.X: batch_x})

            test_error = self.predict(testdata=test_data, gen_error=True)

            print('Predict Generalization Error at {0} : {1} '.format(i, str(test_error)))

        saver = tf.train.Saver()
        save_path = saver.save(self.sess, "./models/autoencoder_processed.ckpt")

        print("Model Saved at Path : " + save_path)

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

    def predict(self, testdata, gen_error = False):

        if gen_error:
            result = self.sess.run([self.loss], feed_dict={self.X: testdata})
        else:
            result = self.sess.run([self.recon_error], feed_dict={self.X: testdata})

        return result[0]

    def predict_non_squarred_error(self, testdata, gen_error = False):

        if gen_error:
            result = self.sess.run([self.loss], feed_dict={self.X: testdata})
        else:
            result = self.sess.run([self.error], feed_dict={self.X: testdata})

        return result[0]