import os
import shutil
import tqdm
from components.metadata_collector import MetadataCollector
from components.feature_extractor import FeatureExtractor
import pandas as pd
from sklearn.model_selection import train_test_split


def pcap2csv(pcap_file_path, dst_csv_path, chunk_size=10000):
    monitor_area = [('192.168.0.0', 2), ('10.0.0.0', 1), ('172.16.0.0', 2)]
    mc = MetadataCollector()
    mc.init_reader(pcap_file_path, monitor_area)
    fe = FeatureExtractor()

    while True:
        time_duration, meta_data_dict = mc.collect_metadata_by_chunk_per_packet_from_pcap(chunk_size=chunk_size)
        if not meta_data_dict:
            break
        samples = fe.extract_features_all(time_duration, meta_data_dict)
        df = pd.DataFrame(samples, columns=['ip', 'sport_uniq_ratio', 'sport_freq_mean', 'sport_freq_var', 'sport_freq_ske',
                                             'sport_freq_kur',
                                             'ip_uniq_ratio', 'ip_freq_mean', 'ip_freq_var', 'ip_freq_ske',
                                             'ip_freq_kur',
                                             'dport_uniq_ratio', 'dport_freq_mean', 'dport_freq_var', 'dport_freq_ske',
                                             'dport_freq_kur',
                                             'time_interval_mean', 'time_interval_var', 'time_interval_ske',
                                             'time_interval_kur',
                                             'bytes_mean', 'bytes_var', 'bytes_ske', 'bytes_kur',
                                             'speed',
                                             'http', 'https', 'ssh', 'telnet', 'mail', 'dns', 'ntp', 'mqtt', 'upnp',
                                             'mysql', 'irc',
                                             'bittorrent'])

        df.to_csv(dst_csv_path, header=(not os.path.exists(dst_csv_path)), index=False, mode='a')


def class2csv(class_dir_path, dst_csv_path, is_tshark=False, chunk_size=10000):
    # If pcaps are in pcapng format, use tshark to transform them into pcap format.
    if is_tshark:
        for root, dirs, files in os.walk(class_dir_path):
            for file in tqdm.tqdm(files):
                if not file.endswith('pcap'):
                    continue
                if file.endswith('ziq_output.pcap'):
                    src_file_path = os.path.join(root, file)
                    shutil.rmtree(src_file_path)
                    continue
                src_file_path = os.path.join(root, file)
                dst_file_path = os.path.join(root, '{}_ziq_output.pcap'.format(file))
                if os.path.exists(dst_csv_path):
                    continue
                os.system('tshark -r {} -F pcap -w {}'.format(src_file_path, dst_file_path))

    ends = 'ziq_output.pcap' if is_tshark is True else 'pcap'
    for root, dirs, files in os.walk(class_dir_path):
        for file in tqdm.tqdm(files):
            print(file)
            if not file.endswith(ends):
                continue
            src_file_path = os.path.join(root, file)
            print(src_file_path)
            pcap2csv(src_file_path, dst_csv_path, chunk_size)


def merge_df_binary(csv1_path, csv2_path, dst_df_path):
    df1 = pd.read_csv(csv1_path)
    df1['label'] = [0] * len(df1)
    df2 = pd.read_csv(csv2_path)
    df2['label'] = [1] * len(df2)
    df = pd.concat([df1, df2], axis=0)
    df = df.sample(frac=1).reset_index(drop=True)
    print(df.head())
    df.to_csv(dst_df_path, header=True, index=False)


def merge_df_multi(df_dir_path, dst_df_path):
    df_list = os.listdir(df_dir_path)
    label_idx = 0
    df_all = pd.DataFrame()
    for df in tqdm.tqdm(df_list):
        df = pd.read_csv(os.path.join(df_dir_path, df))
        df['label'] = [label_idx] * len(df)
        label_idx += 1
        df_all = pd.concat([df_all, df], axis=0)
        df_all = df_all.sample(frac=1).reset_index(drop=True)

    df_all.to_csv(dst_df_path, header=True, index=False)


def merge_df_multi_with_ends(df_dir_path, dst_df_path, end='1000.csv'):
    df_list = os.listdir(df_dir_path)
    df_list_new = []
    for df in df_list:
        if df.endswith(end):
            df_list_new.append(df)
    df_list = df_list_new
    label_idx = 0
    df_all = pd.DataFrame()
    for df in tqdm.tqdm(df_list):
        df = pd.read_csv(os.path.join(df_dir_path, df))
        df['label'] = [label_idx] * len(df)
        label_idx += 1
        df_all = pd.concat([df_all, df], axis=0)
        df_all = df_all.sample(frac=1).reset_index(drop=True)

    df_all.to_csv(dst_df_path, header=True, index=False)


def merge_df_multi_by_dict(df_dir_path, dst_df_path, label_dict):
    df_all = pd.DataFrame()
    for df_name in tqdm.tqdm(label_dict.keys()):
        df = pd.read_csv(os.path.join(df_dir_path, df_name))
        label_idx = label_dict[df_name]
        df['label'] = [label_idx] * len(df)
        df_all = pd.concat([df_all, df], axis=0)
        df_all = df_all.sample(frac=1).reset_index(drop=True)
    df_all.to_csv(dst_df_path, header=True, index=False)


def sample_df(src_df_path, dst_df_path):
    df = pd.read_csv(src_df_path)
    label_set = set(df['label'])
    counts = df['label'].value_counts()
    average_count = int(sum(counts) / len(counts))
    min_count = min(counts)
    print(counts)
    # print(average_count)

    df_sample = pd.DataFrame()
    for label in label_set:
        df_new = df[df['label'] == label].sample(n=average_count, replace=True)
        df_sample = pd.concat([df_sample, df_new], axis=0)

    print(df_sample['label'].value_counts())
    df_sample.to_csv(dst_df_path, header=True, index=False)


def split_dataset(csv_path, dst_dir_path):
    df = pd.read_csv(csv_path)
    df_y = df['label']
    df_x = df.iloc[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.4)

    train_set = pd.concat([X_train, y_train], axis=1)
    test_set = pd.concat([X_test, y_test], axis=1)
    train_set.to_csv('{}/train_set.csv'.format(dst_dir_path), index=False, header=True)
    test_set.to_csv('{}/test_set.csv'.format(dst_dir_path), index=False, header=True)


if __name__ == '__main__':
    dataset_name_list = ['iot-23-gate']
    chunk_size_list = [10000]
    src_dataset_root_path = r'E:\Main\Engineering\PyCharm\Project\IoT\P1DatasetForCompare'
    """
    Dataset directory structure:
    |-src_dataset_root_path
    |--dataset_name
    |----category1
    |------1.pcap
    |------2.pcap
    |----category2
    |------1.pcap
    |------2.pcap
    """

    category_dict = {
        'iot-23-gate': ['benign', 'malicious'],
        'medbiot': ['benign', 'malicious'],
        'ciciot2023': ['benign', 'malicious'],
        'bot-iot': ['scan', 'dos', 'theft']
    }
    for dataset_name in dataset_name_list:
        for chunk_size in chunk_size_list:
            for category in category_dict[dataset_name]:
                """ process """
                src_class_dir_path = r'{}}\{}\{}'.format(src_dataset_root_path, dataset_name, category)
                dst_csv_path = './dataset/chunk/{}/{}_{}.csv'.format(dataset_name, category, chunk_size)
                class2csv(src_class_dir_path, dst_csv_path, chunk_size=chunk_size)

            """ merge """
            csv1_path = './dataset/chunk/{}/benign_{}.csv'.format(dataset_name, chunk_size)
            csv2_path = './dataset/chunk/{}/malicious_{}.csv'.format(dataset_name, chunk_size)
            dst_df_path = './dataset/chunk/{}/df_{}.csv'.format(dataset_name, chunk_size)
            if dataset_name == 'bot-iot':
                continue
            merge_df_binary(csv1_path, csv2_path, dst_df_path)

            """ merge """
            # df_dir_path = './dataset/chunk_experiment/{}'.format(dataset_name)
            # dst_df_path = './dataset/chunk_experiment/{}/df_{}.csv'.format(dataset_name, chunk_size)
            # merge_df_multi_with_ends(df_dir_path, dst_df_path, end='{}.csv'.format(chunk_size))
            """ merge """
            # df_dir_path = './dataset/compare/{}'.format(dataset_name)
            # dst_df_path = './dataset/compare/{}/df.csv'.format(dataset_name)
            # label_dict = {
            #     'ddos.csv': 2,
            #     'dos.csv': 2,
            #     'scan.csv': 1,
            #     'theft.csv': 0,
            # }
            # merge_df_multi_by_dict(df_dir_path, dst_df_path, label_dict)
            """ sample """
            # src_df_path = './dataset/compare/{}/df.csv'.format(dataset_name)
            # dst_df_path = './dataset/compare/{}/df_sample.csv'.format(dataset_name)
            # sample_df(src_df_path, dst_df_path)
            """ split """
            # csv_path = './dataset/compare/{}/df_sample.csv'.format(dataset_name)
            # dst_dir_path = './dataset/compare/{}'.format(dataset_name)
            # split_dataset(csv_path, dst_dir_path)