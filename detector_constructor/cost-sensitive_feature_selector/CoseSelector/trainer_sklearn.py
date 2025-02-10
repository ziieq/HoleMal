import pandas as pd
import model


# def process_field(df, dataset_name):
#     # 整理特征
#     if 'src_ip' in df.columns:
#         df.pop('src_ip')
#     if 'src_port' in df.columns:
#         df.pop('src_port')
#     if 'dst_ip' in df.columns:
#         df.pop('dst_ip')
#     if 'timestamp' in df.columns:
#         df.pop('timestamp')
#
#     df.replace('tcp', 1, inplace=True)
#     df.replace('udp', 0, inplace=True)
#     df.replace('True', 1, inplace=True)
#     df.replace('False', 0, inplace=True)
#
#     df.replace(True, 1, inplace=True)
#     df.replace(False, 0, inplace=True)
#
#     labels = []
#     if 'MedbIot' in dataset_name:
#         labels = ['benign', 'malicious']
#
#     if 'CICIoT2023' in dataset_name:
#         labels = ['Benign', 'BruteForce', 'DDoS', 'Mirai', 'Recon', 'Spoofing', 'Web']
#         df.drop(df[df['label'] == 'Mirai'].index, inplace=True)
#         df.drop(df[df['label'] == 'Web'].index, inplace=True)
#     if 'IoT23' in dataset_name:
#         labels = ['amazon_echo', 'gafgyt', 'hajime', 'hakai', 'hide_and_seek', 'ircbot', 'kenjiro', 'mirai',
#                   'muhstik', 'okiru', 'phillips_hue', 'soomfy_doorlock', 'troii', 'trojan']
#     if 'bot-iot' in dataset_name:
#         labels = ['DOS', 'THEFT', 'SCAN', 'DDOS']
#         #df.drop(df[df['label'] == 'DDOS'].index, inplace=True)
#     label_idx = 0
#     for label in labels:
#         df.replace(label, label_idx, inplace=True)
#         label_idx += 1


def split(df, sample_ratio, feature_list):
    df = pd.DataFrame(df, columns=feature_list + ['label'])
    print(df.head())
    df = df.astype('float32')
    # 打乱
    df_x = df.sample(frac=sample_ratio)
    df_label = df_x.pop('label').astype('float32')
    # print(df_label.value_counts())
    return df_x, df_label


def train_model(model, dataset, feature_list):
    # 定义损失函数、优化器、DataLoader
    df_x, df_label = split(dataset, 1, feature_list)
    model.fit(df_x, df_label)
    return model


def predict(model, dataset, feature_list):
    df_x, df_label = split(dataset, 1, feature_list)
    y_pre = model.predict(df_x)
    return df_label, y_pre


def evaluate_model(model, train_set, test_set, feature_list, is_print=False):
    print('* evaluate : {}'.format(feature_list))
    model = train_model(model, train_set, feature_list)
    df_label, y_pre = predict(model, test_set, feature_list)
    acc, f1, precision, recall, auc, FPR = model.evaluate(df_label, y_pre, print_res=False)
    if is_print:
        print('\n* 验证集评估metric, acc: {}, f1: {}, precision: {}, recall: {}, auc: {}, FPR: {}'.format(acc, f1, precision, recall, auc, FPR))

    return acc, f1, precision, recall, auc, FPR


if __name__ == '__main__':
    """
    ['sport_uniq_ratio', 'sport_freq_mean', 'sport_freq_var', 'sport_freq_ske', 'sport_freq_kur',
                            'ip_uniq_ratio', 'ip_freq_mean', 'ip_freq_var', 'ip_freq_ske', 'ip_freq_kur',
                            'dport_uniq_ratio', 'dport_freq_mean', 'dport_freq_var', 'dport_freq_ske', 'dport_freq_kur',
                            'time_interval_mean', 'time_interval_var', 'time_interval_ske', 'time_interval_kur',
                            'bytes_mean', 'bytes_var', 'bytes_ske', 'bytes_kur',
                            'speed',
                            'http', 'https', 'ssh', 'telnet', 'mail', 'dns', 'ntp', 'mqtt', 'upnp', 'mysql', 'irc',
                            'bittorrent']
    """
    feature_list = ['ip_uniq_ratio', 'dport_uniq_ratio', 'bytes_mean', 'speed', 'http']
    train_set = pd.read_csv('../../dataset/CICIoT2023-1000-1way/train_set.csv')
    eval_set = pd.read_csv('../../dataset/CICIoT2023-1000-1way/test_set.csv')

    fs = model.RandomForest()
    evaluate_model(fs, train_set, eval_set, feature_list, is_print=True)


