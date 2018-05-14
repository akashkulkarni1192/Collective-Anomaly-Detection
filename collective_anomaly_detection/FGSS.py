import pandas as pd
import numpy as np
from sklearn import preprocessing
from collective_anomaly_detection.data_processing import *
from collective_anomaly_detection.DeepAutoEncoderClass import AutoEncoder


class CollectiveAnomalyDetection(object):
    def __init__(self, model):
        self.model = model

    def run(self):
        print("Loading Data...")
        data = read_data()
        print("Loaded the Dataset")
        Xnormal, Ynormal, Xanomaly, Yanomaly = distinguishAnomalousRecords(data)
        print("Distinguished Data")

        # reduce test set
        Xanomaly = self.reduce_dataset(Xanomaly, 10)

        # Split data into Train 1 and Train 2
        T1, T2 = self.split_train_set(Xnormal)
        # Train AutoEncoder on Train 1
        self.model.train(T1, num_steps=2)

        # Train AutoEncoder on Train 2
        train_recon_error = self.model.predict(T2)

        # Create P Value Matrix on Test 2
        test_recon_error = self.model.predict(Xanomaly)
        self.create_p_value_matrix(train_recon_error, test_recon_error)

    def split_train_set(self, trainset):
        mid = int(len(trainset) / 2)

        train_set1 = trainset[:mid, ]
        train_set2 = trainset[mid:, ]
        return train_set1, train_set2

    def create_p_value_matrix(self, train_error, test_error):
        print(train_error.shape[0])
        print(test_error.shape[0])

    def reduce_dataset(self, data, percentage):
        if (percentage >= len(data)):
            cut = len(data)
        else:
            cut = int(percentage / 100 * len(data))
        return data[: cut]


if __name__ == '__main__':
    collective_anomaly_detection = CollectiveAnomalyDetection(AutoEncoder())
    collective_anomaly_detection.run()
