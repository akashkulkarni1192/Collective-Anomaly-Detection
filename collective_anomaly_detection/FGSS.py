import pandas as pd
from collective_anomaly_detection.data_processing import *
from collective_anomaly_detection.DeepAutoEncoderClass import AutoEncoder
import random


def shorten_data(Xnormal, Ynormal, Xanomaly, Yanomaly):
    # Xnormal_pd = pd.DataFrame(Xnormal)
    # Ynormal_pd = pd.DataFrame(Ynormal)
    # Xanomaly_pd = pd.DataFrame(Xanomaly)
    # Yanomaly_pd = pd.DataFrame(Yanomaly)

    return Xnormal[:10], Ynormal[:10], Xanomaly[:2], Yanomaly[:2]


class CollectiveAnomalyDetection(object):
    def __init__(self, model):
        self.model = model

    def run(self, epochs=5):
        data = read_data()
        Xnormal, Ynormal, Xanomaly, Yanomaly = distinguishAnomalousRecords(data)
        Xnormal, Ynormal, Xanomaly, Yanomaly = shorten_data(Xnormal, Ynormal, Xanomaly, Yanomaly)

        T1, T2 = self.split_train_set(Xnormal)

        self.model.train(T1, num_steps=2)
        train_recon_error = self.model.predict(T2)
        test_recon_error = self.model.predict(Xanomaly)

        p_value = self.create_p_value_matrix(train_recon_error, test_recon_error)
        alpha_values = self.get_unique_alpha_values(p_value)
        self.fgss(p_value, alpha_values, epochs)

    def split_train_set(self, trainset):
        mid = int(len(trainset) / 2)
        train_set1 = trainset[:mid, ]
        train_set2 = trainset[mid:, ]
        return train_set1, train_set2

    def create_p_value_matrix(self, train_error, test_error):
        train_error_df = pd.DataFrame(train_error)
        test_error_df = pd.DataFrame(test_error)

        rows, cols = test_error_df.shape

        for i in range(rows):
            for j in range(cols):
                val = train_error_df[j][test_error_df.iloc[i][j] < train_error_df[j]].count()
                test_error_df.iloc[i][j] = val

        return test_error_df.values

    def reduce_dataset(self, data, percentage):
        if (percentage >= len(data)):
            cut = len(data)
        else:
            cut = int(percentage / 100 * len(data))
        return data[: cut]

    def get_unique_alpha_values(self, p_value):
        return np.unique(p_value)

    def select_random(self, vector):
        threshold = random.random()
        selected_cols = []

        for i in range(0, 10):
            random_num = random.random()
            if (random_num >= threshold):
                selected_cols.append(i)
        return selected_cols

    # def compute_N_alpha(self, p_value, alpha):
    #     N_alpha = pd.DataFrame(pd.DataFrame(p_value))
    #     N_alpha =

    def fgss(self, p_value, alpha_values, epoch=2):
        selected_cols = self.select_random(p_value[0])
        print(selected_cols)
        p_value_df = pd.DataFrame(p_value)
        p_value = p_value_df[selected_cols]

        # for alpha in alpha_values:
        #     N_alpha = self.compute_N_alpha(p_value, alpha)
        print("Check")

if __name__ == '__main__':
    # using 2 hidden layers
    collective_anomaly_detection = CollectiveAnomalyDetection(AutoEncoder(10, 5))
    collective_anomaly_detection.run()
    print("Done")
