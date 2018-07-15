import pandas as pd
from collective_anomaly_detection.data_processing import *
from collective_anomaly_detection.DeepAutoEncoderClass import AutoEncoder
from collective_anomaly_detection.result_storer import store_in_excel

import random
import math


def shorten_data(Xnormal, Ynormal, Xanomaly, Yanomaly):
    # return Xnormal[:2000], Ynormal[:2000], Xanomaly[:50], Yanomaly[:50]
    return Xnormal, Ynormal, Xanomaly[:50], Yanomaly[:50]

def split_train_set(trainset):
    mid = int(len(trainset) / 2)
    train_set1 = trainset[:mid, ]
    train_set2 = trainset[mid:, ]
    return train_set1, train_set2

class CollectiveAnomalyDetection(object):
    def __init__(self, model):
        self.model = model

    def run(self, T1, T2, Xanomaly, epochs=5):
        self.model.train(T1, Xanomaly, alpha=0.00001, num_steps=500)
        train_recon_error = self.model.predict(T2)
        test_recon_error = self.model.predict(Xanomaly)

        p_value = self.create_p_value_matrix(train_recon_error, test_recon_error)

        train_recon_error_df = pd.DataFrame(train_recon_error, index=list(range(train_recon_error.shape[0])),
                                            columns=list(range(train_recon_error.shape[1])))
        test_recon_error_df = pd.DataFrame(test_recon_error, index=list(range(test_recon_error.shape[0])),
                                           columns=list(range(test_recon_error.shape[1])))
        store_in_excel(train_recon_error_df, test_recon_error_df, p_value, 'reconstruction_errors', 2)
        # alpha_values = self.get_unique_alpha_values(p_value)
        # self.fgss_wrapper(p_value, alpha_values, epochs)

    def fgss_wrapper(self, p_value, alpha_values, epochs=5):
        for i in range(epochs):
            alpha, records = self.fgss(p_value, alpha_values)
            p_value = p_value[records]
            alpha, labels = self.fgss(p_value.T, alpha_values)
            p_value = p_value.T
            p_value = p_value[labels]
            p_value = p_value.T



    def create_p_value_matrix(self, train_error, test_error):
        train_error_df = pd.DataFrame(train_error)
        test_error_df = pd.DataFrame(test_error, index=list(range(test_error.shape[0])),
                                     columns=list(range(test_error.shape[1])))
        p_value_df = pd.DataFrame(0, index=list(range(test_error.shape[0])),
                                  columns=list(range(test_error.shape[1])))
        rows, cols = test_error_df.shape

        for i in range(rows):
            for j in range(cols):
                val = train_error_df[j][test_error_df.iloc[i][j] < train_error_df[j]].count()
                p_value_df.iloc[i][j] = val

        return p_value_df

    def reduce_dataset(self, data, percentage):
        if (percentage >= len(data)):
            cut = len(data)
        else:
            cut = int(percentage / 100 * len(data))
        return data[: cut]

    def get_unique_alpha_values(self, p_value):
        return np.unique(p_value)

    def select_random(self, vector):
        # threshold = random.random()
        threshold = 999999.0
        selected_cols = []

        for col in vector:
            random_num = random.random()
            if (random_num >= threshold):
                selected_cols.append(col)
        if len(selected_cols) == 0:
            # selected_cols = self.select_random(vector)
            selected_cols.append(vector[0])
        return selected_cols

    def compute_N_alpha(self, p_value, alpha):
        N_alpha = pd.DataFrame(0, columns=list(p_value.columns.values), index=list(p_value.index.values))

        for i, row in p_value.iterrows():
            for j, value in row.iteritems():
                if value > alpha:
                    N_alpha.iloc[i][j] = 1
                else:
                    N_alpha.iloc[i][j] = 0

        return pd.DataFrame(N_alpha)

    def obtain_subset_with_max_score(self, sorted_index, N_alpha_sum, alpha, n_cols):
        subset_score = 0
        max_score = 0
        end_index = 0
        for index in sorted_index:
            subset_score = subset_score + N_alpha_sum[index]
            ideal_score = (index + 1) * n_cols * alpha
            numerator = subset_score - ideal_score
            val = subset_score * ideal_score
            if (subset_score * ideal_score > 0):
                denominator = math.sqrt(subset_score * ideal_score)
                calculated_score = numerator / denominator
            else:
                calculated_score = 0
            if (calculated_score > 0 and calculated_score > subset_score):
                max_score = calculated_score
                end_index = index
        return sorted_index[: end_index + 1], max_score

    def fgss(self, p_value, alpha_values):
        dropping_cols = self.select_random(list(p_value.columns.values))
        print(dropping_cols)

        selected_cols = list(set(list(p_value.columns.values)) - set(dropping_cols))

        p_value = p_value.drop(dropping_cols, axis=1)
        best_alpha = 0
        best_score = -1
        best_subset = []
        for alpha in alpha_values:
            N_alpha = self.compute_N_alpha(p_value, alpha)
            N_alpha_sum = N_alpha.sum(axis=1).values
            sorted_index = np.argsort(N_alpha_sum)
            subset_index, score = self.obtain_subset_with_max_score(sorted_index, N_alpha_sum, alpha,
                                                                    len(selected_cols))
            if (score > best_score):
                best_alpha = alpha
                best_score = score
                best_subset = subset_index

        return best_alpha, best_subset


if __name__ == '__main__':
    data = read_data()
    Xnormal, Ynormal, Xanomaly, Yanomaly = distinguishAnomalousRecords(data)
    # Xnormal, Ynormal, Xanomaly, Yanomaly = shorten_data(Xnormal, Ynormal, Xanomaly, Yanomaly)

    T1, T2 = split_train_set(Xnormal)


    # using 2 hidden layers
    collective_anomaly_detection = CollectiveAnomalyDetection(AutoEncoder(10, 5, Xanomaly))
    collective_anomaly_detection.run(T1=T1, T2=T2, Xanomaly= Xanomaly)
    print("Done")
