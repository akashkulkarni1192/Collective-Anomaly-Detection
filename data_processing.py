from sklearn import preprocessing
import pandas as pd
import numpy as np
import os

protocol_map = {'tcp': 1, 'udp': 2, 'icmp': 3}
service_map = {'service': 1, 'http': 2, 'smtp': 3, 'finger': 4, 'domain_u': 5, 'auth': 6, 'telnet': 7, 'ftp': 8,
               'eco_i': 9,
               'ntp_u': 10, 'ecr_i': 11, 'other': 12, 'private': 13, 'pop_3': 14, 'ftp_data': 15, 'rje': 16, 'time': 17,
               'mtp': 18,
               'link': 19, 'remote_job': 20, 'gopher': 21, 'ssh': 22, 'name': 23, 'whois': 24, 'domain': 25,
               'login': 26, 'imap4': 27,
               'daytime': 28, 'ctf': 29, 'nntp': 30, 'shell': 31, 'IRC': 32, 'nnsp': 33, 'http_443': 34, 'exec': 35,
               'printer': 36,
               'efs': 37, 'courier': 38, 'uucp': 39, 'klogin': 40, 'kshell': 41, 'echo': 42, 'discard': 43,
               'systat': 44,
               'supdup': 45, 'iso_tsap': 46, 'hostnames': 47, 'csnet_ns': 48, 'pop_2': 49, 'sunrpc': 50,
               'uucp_path': 51, 'ldap': 52, 'netstat': 53, 'urh_i': 54, 'X11': 55, 'urp_i': 56, 'pm_dump': 57,
               'tftp_u': 58, 'tim_i': 59,
               'red_i': 60, 'netbios_ns': 61, 'netbios_ssn': 62, 'netbios_dgm': 63, 'sql_net': 64, 'vmnet': 65,
               'bgp': 66, 'Z39_50': 67}

flag_map = {'flag': 1, 'SF': 2, 'S1': 3, 'REJ': 4, 'S2': 5, 'S0': 6, 'S3': 7, 'RSTO': 8, 'RSTR': 9, 'RSTOS0': 10,
            'OTH': 11, 'SH': 12}


def read_data():
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
    kdd_data_10percent = pd.read_csv("./data/kddcup_10_percent_corrected_processed_new.csv", header=None, names=col_names,
                                     dtype=object)

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
    # data = kdd_data_10percent[num_features]
    # data = np.array(data)
    # data = normalize_data(data)
    kdd_data_10percent = kdd_data_10percent.iloc[1:, :]
    return kdd_data_10percent


def normalize_data(X):
    X = preprocessing.normalize(X, axis=1)
    return X


def distinguishAnomalousRecords(data):
    normalRecords = data[(data["label"] == "normal.")]
    anoRecords = data[(data["label"] != "normal.")]
    nR = np.array(normalRecords)[:, 0:-1].astype(float)
    nL = np.array(normalRecords)[:, -1]
    aR = np.array(anoRecords)[:, 0:-1].astype(float)
    aL = np.array(anoRecords)[:, -1]
    return normalize_data(nR), nL, normalize_data(aR), aL


def process():
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
    unprocessed_data = pd.read_csv("./data/kddcup_10_percent_corrected_processed_new.csv", header=None, names=col_names,
                                   dtype=object)

    protocol_types = unprocessed_data["protocol_type"].unique()  # returns 3 types ['tcp', 'udp', 'icmp']
    unprocessed_data.loc[unprocessed_data['protocol_type'] == 'tcp', 'protocol_type'] = protocol_map['tcp']
    unprocessed_data.loc[unprocessed_data['protocol_type'] == 'udp', 'protocol_type'] = protocol_map['udp']
    unprocessed_data.loc[unprocessed_data['protocol_type'] == 'icmp', 'protocol_type'] = protocol_map['icmp']

    service_types = unprocessed_data['service'].unique()

    for service_type in service_map:
        unprocessed_data.loc[unprocessed_data['service'] == service_type, 'service'] = service_map[service_type]
    print(str(service_types))

    flag_types = unprocessed_data['flag'].unique()
    print(str(flag_types))

    for flag_type in flag_map:
        unprocessed_data.loc[unprocessed_data['flag'] == flag_type, 'flag'] = flag_map[flag_type]

    print("Writing csv file")
    # unprocessed_data.to_csv('./data/kddcup_10_percent_corrected_processed_new.csv', )
    return None

# process()
