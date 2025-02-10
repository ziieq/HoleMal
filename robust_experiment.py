import os.path
from sklearn.model_selection import train_test_split
import pandas as pd
from HoleMal.components.classifier.ml.rf import RandomForest
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def evaluate(y_test, y_pre, print_res=False):
    if print_res:
        print()
    acc = accuracy_score(y_test, y_pre)
    if print_res:
        print('acc: ', acc)

    f1 = f1_score(y_test, y_pre, average='macro')
    if print_res:
        print('f1: ', f1)

    recall = recall_score(y_test, y_pre, average='macro')
    if print_res:
        print('recall: ', recall)

    precision = precision_score(y_test, y_pre, average='macro')
    if print_res:
        print('precision: ', precision)
    if print_res:
        print()

    """
    FPR
    """
    cnf_matrix = confusion_matrix(y_test, y_pre)
    print(cnf_matrix)
    # [[1 1 3]
    # [3 2 2]
    # [1 3 1]]

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    FPR = FP / (FP + TN)
    return acc, f1, recall, precision, FPR


def sample_df(df):
    label_set = set(df['label'])
    counts = df['label'].value_counts()
    average_count = int(sum(counts) / len(counts))
    print(counts)
    df_sample = pd.DataFrame()
    for label in label_set:
        df_new = df[df['label'] == label].sample(n=average_count, replace=True)
        df_sample = pd.concat([df_sample, df_new], axis=0)
    print(df_sample['label'].value_counts())
    return df_sample


def split_dataset(df):
    df_y = df['label']
    df_x = df.iloc[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.4)
    train_set = pd.concat([X_train, y_train], axis=1)
    test_set = pd.concat([X_test, y_test], axis=1)
    return train_set, test_set


def split(df, sample_ratio):
    df = df.astype('float32')
    # 打乱
    df_x = df.sample(frac=sample_ratio)
    df_label = df_x.pop('label').astype('float32')
    # print(df_label.value_counts())
    return df_x, df_label
def train_model_ml(model, dataset):
    # 定义损失函数、优化器、DataLoader
    df_x, df_label = split(dataset, 1)
    model.fit(df_x, df_label)
    return model

def predict_ml(model, dataset):
    df_x, df_label = split(dataset, 1)
    y_pre = model.predict(df_x)
    return df_label, y_pre

def evaluate_model_ml(model, train_set, test_set, is_print=False):
    model = train_model_ml(model, train_set)
    df_label, y_pre = predict_ml(model, test_set)
    acc, f1, precision, recall, FPR = evaluate(df_label, y_pre, print_res=False)
    if is_print:
        print('\n* 验证集评估metric, acc: {}, f1: {}, precision: {}, recall: {}, FPR: {}'.format(acc, f1, precision, recall, FPR))
    print(len(test_set))
    return acc, f1, precision, recall, FPR


if __name__ == '__main__':
    dataset_name = 'iot-23-gate'
    feature_subset_dict = {
        'iot-23-gate': ['speed', 'bytes_mean'],  # ['speed', 'https']
        'medbiot': ['bytes_mean', 'speed'],
        'ciciot2023': ['ip_uniq_ratio', 'ip_freq_var', 'time_interval_kur', 'speed'],
        'bot-iot': ['sport_uniq_ratio', 'dport_freq_mean', 'bytes_mean', 'speed']
    }

    feature_subset = feature_subset_dict[dataset_name]
    rf = RandomForest()
    data = {
        'acc': [],
        'f1': [],
        'precision': [],
        'recall': [],
        'FPR': []   # [[1, 1],[2, 2],[3, 3],[4, 4]]
    }
    test_rate_list = ['02', '04', '06', '08']
    for test_rate in test_rate_list:
        for _ in range(10):
            train_set = pd.read_csv('./dataset/robust/{}/df_10000_00.csv'.format(dataset_name))
            train_set = pd.DataFrame(train_set, columns=feature_subset+['label'])

            test_set = pd.read_csv('./dataset/robust/{}/df_10000_{}.csv'.format(dataset_name, test_rate))
            test_set = pd.DataFrame(test_set, columns=feature_subset+['label'])

            acc, f1, precision, recall, FPR = evaluate_model_ml(rf, train_set, test_set, is_print=True)
            data['acc'].append(acc)
            data['f1'].append(f1)
            data['precision'].append(precision)
            data['recall'].append(recall)
            data['FPR'].append(FPR)

        data_all = {'dataset': [dataset_name], 'test_rate': [test_rate]}
        for key, value in data.items():
            if key == 'FPR':
                L = data['FPR']
                L = list(map(list, zip(*L)))
                L = [sum(x)/len(x) for x in L]
                data_all['FPR'] = [str(L)]
            else:
                data_all[key] = [sum(data[key])/len(data[key])]

        df_res = pd.DataFrame(data_all)
        res_path = './res_robust.csv'
        df_res.to_csv(res_path, header=(not os.path.exists(res_path)), index=False, mode='a')
