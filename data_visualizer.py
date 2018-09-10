from result_storer import load_from_excel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class DataVisualizer(object):
    def plot_histogram(self, data, title):
        # print(data.shape)
        data = np.array(data)

        buckets = pd.cut(data, 10)
        print(buckets.value_counts())
        # buckets.value_counts().plot.bar(rot=0, color="b", figsize=(20, 4))

        plt.hist(data)

        plt.title(title)
        plt.show()

    def plot_scatter_graph(self, data):
        y = np.array(data)
        x = range(len(data))
        plt.plot(x, y, 'ro')
        plt.show()

    def correlation(self, data):
        corr = data.corr()
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        sns.heatmap(corr, mask=mask,
                    xticklabels=corr.columns.values,
                    yticklabels=corr.columns.values)
        plt.show()

        # fig, ax = plt.subplots(figsize=(20, 20))
        # ax.matshow(corr)
        # plt.xticks(range(len(corr.columns)), corr.columns)
        # plt.yticks(range(len(corr.columns)), corr.columns)
        # plt.show()

    def filter_data(self, data):
        print("Filtering 0s and 1s ...")
        result = []

        for val in data:
            if val > 0 and val < 1:
                result.append(val)
        return result

def analyze_results():
    data_visualizer = DataVisualizer()
    train_recon, test_recon, train_p_value, test_p_value = load_from_excel('reconstruction_errors-non_sq.xlsx')

    # data_visualizer.plot_scatter_graph(train_recon[28])

    # train_recon = data_visualizer.filter_columns(train_recon, [0, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26, 27, 29, 34, 37, 38, 39, 40])
    data_visualizer.plot_histogram(data_visualizer.filter_data(train_recon[1]), 'Train Set Reconstruction errors of one column (feature)')
    data_visualizer.plot_histogram(data_visualizer.filter_data(train_recon.iloc[28]), 'Train Set Reconstruction errors of one row (record)')
    multi_rows = list(range(10, 26))
    multi_cols = list(range(1,6)) + [11] + [22, 23] + [28] + list(range(30, 34)) + [35, 36]
    submatrix = train_recon.iloc[multi_rows][multi_cols]
    data_visualizer.plot_histogram(data_visualizer.filter_data(submatrix.values.flatten()),
                                   'Train Set Reconstruction errors of multiple row/columns')

    data_visualizer.plot_histogram(data_visualizer.filter_data(test_recon[28]), 'Test Set Reconstruction errors of one column (feature)')
    data_visualizer.plot_histogram(data_visualizer.filter_data(test_recon.iloc[28]), 'Test Set Reconstruction errors of one row (record)')
    multi_rows = list(range(10, 26))
    multi_cols = list(range(1, 5)) + [22, 23] + [28] + list(range(31, 34)) + [35]
    submatrix = test_recon.iloc[multi_rows][multi_cols]
    data_visualizer.plot_histogram(data_visualizer.filter_data(submatrix.values.flatten()),
                                   'Test Set Reconstruction errors of multiple row/columns')


    data_visualizer.plot_histogram(data_visualizer.filter_data(train_p_value[28]), 'Train P value of one column (feature)')
    data_visualizer.plot_histogram(data_visualizer.filter_data(train_p_value.iloc[28]), 'Train P value of one row (record)')
    multi_rows = list(range(10, 26))
    multi_cols = list(range(1, 6)) + [11] + [22, 23] + [28] + list(range(31, 34)) + [36]
    submatrix = train_p_value.iloc[multi_rows][multi_cols]
    submatrix = train_p_value
    data_visualizer.plot_histogram(data_visualizer.filter_data(submatrix.values.flatten()),
                                   'TrainP value of multiple row/columns')

    data_visualizer.plot_histogram(data_visualizer.filter_data(test_p_value[28]), 'Test P value of one column (feature)')
    data_visualizer.plot_histogram(data_visualizer.filter_data(test_p_value.iloc[28]), 'Test P value of one row (record)')
    multi_rows = list(range(10, 26))
    multi_cols = list(range(1, 4)) + [28] + list(range(31, 34)) + [35, 36]
    submatrix = test_p_value.iloc[multi_rows][multi_cols]
    submatrix = test_p_value
    data_visualizer.plot_histogram(data_visualizer.filter_data(submatrix.values.flatten()),
                                   'Test P value of multiple row/columns')

    analyze_corr = train_recon.iloc[15:30][0:6]
    data_visualizer.correlation(analyze_corr)


analyze_results()
