import pandas as pd
import numpy as np
from sklearn import preprocessing

def normalize_data(X):
    X = preprocessing.normalize(X, axis=1)
    return X

def get_data():
    col_names = ["duration", "protocol_type", "service", "flag", "src_bytes",
                 "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
                 "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
                 "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
                 "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
                 "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                 "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                 "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                 "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                 "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]
    kdd_data_10percent = pd.read_csv("./data/kddcup.data_10_percent_corrected", header=None, names=col_names)
    num_features = [
        "duration", "src_bytes",
        "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
        "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
        "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
        "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
        "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
    ]

    # labels = kdd_data_10percent['label'].copy()
    # labels[labels != 'normal.'] = 'attack.'
    # labels.value_counts()
    # data = kdd_data_10percent[num_features].astype(float)
    data = kdd_data_10percent[num_features]
    # data = np.array(data)
    # data = normalize_data(data)
    return data

def distinguishAnomalousRecords(data):
    normalRecords = data[(data["label"] == "normal.")]
    anoRecords = data[(data["label"] != "normal.")]
    nR = np.array(normalRecords)[:, 0:-1].astype(float)
    nL = np.array(normalRecords)[:, -1]
    aR = np.array(anoRecords)[:, 0:-1].astype(float)
    aL = np.array(anoRecords)[:, -1]
    return normalize_data(nR), nL, normalize_data(aR), aL

if __name__ == '__main__':
    print("Loading Data...")
    data = get_data()
    print("Loaded the Dataset")
    Xnormal, Ynormal, Xanomaly, Yanomaly = distinguishAnomalousRecords(data)
    print("Distinguished Data")