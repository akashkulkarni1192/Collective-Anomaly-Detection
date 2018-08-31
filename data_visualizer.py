from result_storer import load_from_excel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class DataVisualizer(object):
    def plot_bar_graph(self, data, title):
        print(data.shape)
        data = np.array(data)
        buckets = pd.cut(data, 10)

        print(buckets.value_counts())
        buckets.value_counts().plot.bar(rot=0, color="b", figsize=(20, 4))
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


def analyze_results():
    data_visualizer = DataVisualizer()
    train_recon, test_recon, train_p_value, test_p_value = load_from_excel('reconstruction_errorsprocessed.xlsx')

    # data_visualizer.plot_scatter_graph(train_recon[28])

    data_visualizer.plot_bar_graph(train_recon[28], 'Train Set Reconstruction errors of one column (feature)')
    data_visualizer.plot_bar_graph(train_recon.iloc[28], 'Training Reconstruction errors of one row (record)')
    submatrix = train_recon.iloc[10:26][1:5]
    data_visualizer.plot_bar_graph(np.array(submatrix).reshape((submatrix.shape[0] * submatrix.shape[1])),
                                   'Train Set Reconstruction errors of 16 records and 5 features')

    data_visualizer.plot_bar_graph(test_recon[28], 'Test Set Reconstruction errors of one column (feature)')
    data_visualizer.plot_bar_graph(test_recon.iloc[28], 'Test Set Reconstruction errors of one row (record)')
    submatrix = test_recon.iloc[10:26][1:5]
    data_visualizer.plot_bar_graph(np.array(submatrix).reshape((submatrix.shape[0] * submatrix.shape[1])),
                                   'Test Set Reconstruction errors of 16 records and 5 features')

    index = 28
    res = train_p_value[index]
    data_visualizer.plot_bar_graph(res, 'Train P value of one column (feature)')
    data_visualizer.plot_bar_graph(train_p_value.iloc[index], 'Train P value of one row (record)')
    submatrix = train_p_value.iloc[15:30]
    data_visualizer.plot_bar_graph(np.array(submatrix).reshape((submatrix.shape[0] * submatrix.shape[1])),
                                   'TrainP value of 15 rows (records)')

    res = test_p_value[index]
    data_visualizer.plot_bar_graph(res, 'Test P value of one column (feature)')
    data_visualizer.plot_bar_graph(test_p_value.iloc[index], 'Test P value of one row (record)')
    submatrix = test_p_value.iloc[15:30]
    data_visualizer.plot_bar_graph(np.array(submatrix).reshape((submatrix.shape[0] * submatrix.shape[1])),
                                   'Test P value of 15 rows (records)')

    analyze_corr = train_recon.iloc[15:30][0:6]
    data_visualizer.correlation(analyze_corr)


analyze_results()
