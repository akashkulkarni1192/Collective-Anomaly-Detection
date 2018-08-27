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

        buckets.value_counts().plot.bar(rot=0, color="b", figsize=(20, 4))
        plt.title(title)
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
    train_recon, test_recon, p_value = load_from_excel('reconstruction_errors100.xlsx')

    # data_visualizer.plot_bar_graph(p_value[28], 'column')
    # data_visualizer.plot_bar_graph(p_value.iloc[0], 'row')
    submatrix = p_value.iloc[:6]
    # data_visualizer.plot_bar_graph(np.array(submatrix).reshape((submatrix.shape[0] * submatrix.shape[1])), 'row-column')
    analyze_corr = train_recon.iloc[:10]
    data_visualizer.correlation(analyze_corr)

