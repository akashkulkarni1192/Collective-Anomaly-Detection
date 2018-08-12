from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas

def normalize_data(X):
    X = preprocessing.normalize(X, axis=1)
    return X


col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
kdd_data_10percent = pandas.read_csv("./data/kddcup.data_10_percent_corrected", header=None, names = col_names)
num_features = [
    "duration","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate"
]

features = kdd_data_10percent[num_features].astype(float)
# features = process_data/(features)
features = normalize_data(features)
# np.random.shuffle(features)
# Training Parameters
learning_rate = 0.01
num_steps = 100
batch_size = 20

display_step = 100


# Network Parameters
num_hidden_1 = 15 # 1st layer num features
num_input = features.shape[1] # MNIST data input (img shape: 28*28)
LOG_DIR = '/tmp/tensorflow_auto_encoder/deepae13'
n_batches = int(features.shape[0] / batch_size)
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b1': tf.Variable(tf.random_normal([num_input])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    # layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
    #                                biases['encoder_b2']))
    return layer_1


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    # layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
    #                                biases['decoder_b2']))
    return layer_1

# Construct models
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
# with tf.name_scope("loss"):
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))


# with tf.name_scope("train"):
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
tf.summary.scalar('loss', loss)
merged_summary_op = tf.summary.merge_all()

# summ = tf.summary.merge_all()
# Start Training
# Start a new TF session
summary_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())
with tf.Session() as sess:

    # Run the initializer
    # saver = tf.train.Saver()
    sess.run(init)
    # writer = tf.summary.FileWriter(LOG_DIR+'lossGraph')
    # writer.add_graph(sess.graph)
    # Training
    for i in range(1, num_steps+1):
        for j in range(n_batches):
            batch_x = features[j * batch_size:(j * batch_size + batch_size), ]

            _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
            summary = sess.run(merged_summary_op, feed_dict={X: batch_x})

            # if j % display_step == 0 or i == 1:
            #     print('Step %i: Batch: %i Minibatch Loss: %f' % (i, j, l))
        if i % 1 == 0:
                    print('Step %i: Minibatch Loss: %f' % (i, l))
        summary_writer.add_summary(summary, i)
