import pandas
import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss

# Must declare data_dir as the directory of training and test files

train_data = "./data/kddcup.data_10_percent_corrected"
# train_labels = data_dir + "train_labels.txt"
# test_data = data_dir + "corrected"
# test_labels = data_dir + "test_labels.txt"


def process_data(X):
    X = X.drop(41, 1)
    X[1], uniques = pandas.factorize(X[1])
    X[2], uniques = pandas.factorize(X[2])
    X[3], uniques = pandas.factorize(X[3])

    num_examples = 10 ** 6
    X = X[0:num_examples]
    # y = y[0:num_examples]

    X = numpy.array(X)
    # y = numpy.array(y).ravel()

    return X


print("Loading training data")
train_X = pandas.read_csv(train_data, header=None)
# train_y = pandas.read_csv(train_labels, header=None)
train_X = process_data(train_X)

print(train_X.shape)

# print("Training and predicting")
# learner = KNeighborsClassifier(1, n_jobs=-1)
# learner.fit(train_X, train_y)
# pred_y = learner.predict(test_X)
#
# results = confusion_matrix(test_y, pred_y)
# error = zero_one_loss(test_y, pred_y)

# print(results)
# print(error)