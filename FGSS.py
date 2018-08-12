from data_processing import *
from DeepAutoEncoderClass import AutoEncoder
from result_storer import store_in_excel

import random
import math


def shorten_data(Xnormal, Ynormal, Xanomaly, Yanomaly):
    # return Xnormal[:2000], Ynormal[:2000], Xanomaly[:50], Yanomaly[:50]
    return Xnormal[:200], Ynormal[:200], Xanomaly[:50], Yanomaly[:50]


def split_train_set(trainset):
    mid = int(len(trainset) / 2)
    train_set1 = trainset[:mid, ]
    train_set2 = trainset[mid:, ]
    return train_set1, train_set2


class CollectiveAnomalyDetection(object):
    def __init__(self, model):
        self.model = model

    def run(self, T1, T2, Xanomaly, cache = False, epochs=5):
        if cache == False:
            self.model.train(T1, Xanomaly, alpha=0.00001, num_steps=20000)

        print("Creating Training Reconstruction error...")
        train_recon_error = self.model.predict(T2)
        print("Creating Test Reconstruction error...")
        test_recon_error = self.model.predict(Xanomaly)
        print("Creating P Value...")
        p_value = self.create_p_value_matrix(train_recon_error, test_recon_error)

        train_recon_error_df = pd.DataFrame(train_recon_error, index=list(range(train_recon_error.shape[0])),
                                            columns=list(range(train_recon_error.shape[1])))
        test_recon_error_df = pd.DataFrame(test_recon_error, index=list(range(test_recon_error.shape[0])),
                                           columns=list(range(test_recon_error.shape[1])))
        store_in_excel(train_recon_error_df, test_recon_error_df, p_value, 'reconstruction_errors', 0)
        # alpha_values = self.get_unique_alpha_values(p_value)
        # self.fgss_wrapper(p_value, epochs)

    def fgss_wrapper(self, master_p_value, epochs=5):
        global_rows, global_cols, global_alpha, global_score = None, None, None, -1

        for epoch in range(epochs):

            prev_rows, prev_cols, prev_alpha, prev_score = None, None, None, -1

            while (True):
                # Considering all rows, selective columns -> get selected rows
                p_value = master_p_value.copy()
                dropping_cols = self.select_random(list(p_value.columns.values))
                rows, cols, alpha, score = self.fgsss(p_value, dropping_cols)
                if (score < prev_score):
                    break
                else:
                    prev_score = score
                    prev_rows = rows
                    prev_cols = cols
                    prev_alpha = alpha

                # Considering all columns, selective rows -> get selected columns
                p_value = master_p_value.T.copy()
                dropping_cols = self.select_random(list(p_value.columns.values))
                cols, rows, alpha, score = self.fgsss(p_value, dropping_cols)
                if (score < prev_score):
                    break
                else:
                    prev_score = score
                    prev_rows = rows
                    prev_cols = cols
                    prev_alpha = alpha

                print(score)
            if (prev_score > global_score):
                global_score = prev_score
                global_rows = prev_rows
                global_cols = prev_cols
                global_alpha = prev_alpha

        return global_rows, global_cols, global_alpha, global_score

    def fgsss(self, p_value, drop_vector):
        p_value = p_value.drop(drop_vector, axis=1)
        alpha_values = self.get_unique_alpha_values(p_value)

        best_alpha = 0
        best_score = -1
        best_subset = []
        for alpha in alpha_values:
            N_alpha = self.compute_N_alpha(p_value, alpha)
            N_alpha_sum = N_alpha.sum(axis=1).values
            sorted_index = np.argsort(N_alpha_sum)
            subset_index, score = self.obtain_subset_with_max_score(sorted_index, N_alpha_sum, alpha,
                                                                    p_value.columns.values.size)
            if (score > best_score):
                best_alpha = alpha
                best_score = score
                best_subset = subset_index

        return best_subset, drop_vector, best_alpha, best_score

    def create_p_value_matrix(self, train_error, test_error):
        train_error_df = pd.DataFrame(train_error)
        test_error_df = pd.DataFrame(test_error, index=list(range(test_error.shape[0])),
                                     columns=list(range(test_error.shape[1])))
        p_value_df = pd.DataFrame(0, index=list(range(test_error.shape[0])),
                                  columns=list(range(test_error.shape[1])))
        rows, cols = test_error_df.shape
        rows_train, cols_train = train_error_df.shape
        for i in range(rows):
            for j in range(cols):
                # val = train_error_df[j]
                val = train_error_df[j][test_error_df.iloc[i][j] <= train_error_df[j]].count()
                val = val / float(rows_train)
                p_value_df.loc[i, j] = val

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
        max_score = -1
        max_score = 0
        end_index = 0
        subset_score = 0
        for index in sorted_index:
            subset_score = subset_score + N_alpha_sum[index]
            ideal_score = (index + 1) * n_cols * alpha
            if subset_score < ideal_score:
                calculated_score = 0
            else:
                numerator = subset_score - ideal_score
                if (ideal_score > 0):
                    denominator = math.sqrt(ideal_score * (1 - alpha))
                    if denominator > 0:
                        calculated_score = numerator / denominator
                    else:
                        calculated_score = 0
                else:
                    calculated_score = 0
            if (calculated_score > 0 and calculated_score > max_score):
                max_score = calculated_score
                end_index = index
        return sorted_index[: end_index + 1], max_score

    # def fgss(self, p_value, alpha_values):
    #     dropping_cols = self.select_random(list(p_value.columns.values))
    #     print(dropping_cols)
    #
    #     selected_cols = list(set(list(p_value.columns.values)) - set(dropping_cols))
    #
    #     p_value = p_value.drop(dropping_cols, axis=1)
    #     best_alpha = 0
    #     best_score = -1
    #     best_subset = []
    #     for alpha in alpha_values:
    #         N_alpha = self.compute_N_alpha(p_value, alpha)
    #         N_alpha_sum = N_alpha.sum(axis=1).values
    #         sorted_index = np.argsort(N_alpha_sum)
    #         subset_index, score = self.obtain_subset_with_max_score(sorted_index, N_alpha_sum, alpha,
    #                                                                 len(selected_cols))
    #         if (score > best_score):
    #             best_alpha = alpha
    #             best_score = score
    #             best_subset = subset_index
    #
    #     return best_alpha, best_subset


def main():
    data = read_data()
    Xnormal, Ynormal, Xanomaly, Yanomaly = distinguishAnomalousRecords(data)
    Xnormal, Ynormal, Xanomaly, Yanomaly = shorten_data(Xnormal, Ynormal, Xanomaly, Yanomaly)

    T1, T2 = split_train_set(Xnormal)

    # using 2 hidden layers
    use_cached_model = True
    collective_anomaly_detection = CollectiveAnomalyDetection(AutoEncoder(10, 5, Xanomaly, use_cached_model))
    collective_anomaly_detection.run(T1=T1, T2=T2, Xanomaly=Xanomaly, cache=use_cached_model)
    print("Done")

main()