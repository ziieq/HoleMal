from sklearn.model_selection import train_test_split
import pandas as pd
from HoleMal.components.classifier.ml.rf import RandomForest
from HoleMal.components.classifier.ml.svm import SVM
from HoleMal.components.classifier.ml.lr import LogisticR
from HoleMal.components.classifier.ml.knn import KNN
from HoleMal.components.classifier.ml.lgb import LGB
from HoleMal.components.classifier.ml.tree import Tree
from HoleMal.components.classifier.dl.gru import GRU
from HoleMal.components.classifier.dl.mlp import MLP
from line_profiler import LineProfiler
from functools import wraps
import tracemalloc
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import sys
import os
import torch.nn as nn
import torch.optim as optim
from HoleMal.components.classifier.dl.dataset_iot import DatasetIoT
from torch.utils.data import DataLoader
import torchmetrics


class Redirect:
    """
    将cmd输出捕获到变量里，获取line_profiler的输出。
    """
    def __init__(self):
        self.backup = sys.stdout
        self.content = ''

    def write(self, s):
        self.content += s

    def transform(self):
        sys.stdout = self

    def restore(self):
        sys.stdout = self.backup


def func_line_time(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        # func_return = f(*args, **kwargs)
        lp = LineProfiler()
        lp_wrap = lp(f)
        lp_wrap(*args, **kwargs)
        lp.print_stats()
        # return func_return
    return decorator


"""
Metric Score:
"""

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

    return acc, f1, recall, precision


def sample_df(df):
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
    return df_sample


"""
DataFrame Split:
"""

def split_dataset(df):
    df = df.sample(frac=1)
    df_y = df['label']
    df_x = df.iloc[:, :-1].astype('float32')
    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.4)
    train_set = pd.concat([X_train, y_train], axis=1)
    test_set = pd.concat([X_test, y_test], axis=1)
    return train_set, test_set


"""
Machine Learning methods:
"""

def train_model_ml(model, dataset, feature_list):
    # 定义损失函数、优化器、DataLoader
    df_label = dataset.iloc[:, -1]
    df_x = dataset.iloc[:, :-1]
    model.fit(df_x, df_label)
    return model

def predict_ml(model, dataset, feature_list):
    df_label = dataset.iloc[:, -1]
    df_x = dataset.iloc[:, :-1]
    tracemalloc.start()
    y_pre = model.predict(df_x)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return df_label, y_pre, peak / 10 ** 6

@func_line_time
def predict_ml_for_time(model, dataset, feature_list):
    model.predict(dataset.iloc[:, :-1])

def evaluate_model_ml(model, train_set, test_set, feature_list, is_print=False):
    print('* evaluate : {}'.format(feature_list))
    model = train_model_ml(model, train_set, feature_list)
    redirect = Redirect()
    redirect.transform()
    predict_ml_for_time(model, test_set, feature_list)
    redirect.restore()
    line_time_list = redirect.content.split('\n')
    cal_time = 1
    for line in line_time_list:
        if 'model.predict(' in line:
            line_not_none = []
            for x in line.split(' '):
                if x != '':
                    line_not_none.append(x)
            cal_time = float(line_not_none[2]) / (10 ** 7)

    df_label, y_pre, peak = predict_ml(model, test_set, feature_list)
    acc, f1, precision, recall = evaluate(df_label, y_pre, print_res=False)
    if is_print:
        print('\n* 验证集评估metric, acc: {}, f1: {}, precision: {}, recall: {}'.format(acc, f1, precision, recall))
    print(len(test_set))
    return f1, peak, cal_time, len(test_set) / cal_time


"""
Deep Learning methods:
"""

@func_line_time
def evaluate_model_dl_for_time(model, train_set, test_set, feature_list, is_print=False):
    predict_dl(model, test_set, feature_list)

def train(model, train_set, epochs=20):

    # 定义损失函数、优化器、DataLoader
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataloader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)

    acc = torchmetrics.Accuracy(task='multiclass', num_classes=2, average='micro')
    f1 = torchmetrics.F1Score(task='multiclass', num_classes=2, average='macro')
    recall = torchmetrics.Recall(task='multiclass', num_classes=2, average='macro')
    precision = torchmetrics.Precision(task='multiclass', num_classes=2, average='macro')

    # 开始训练
    for epoch in range(epochs):
        for i, (labels, features) in enumerate(dataloader):
            print('\r当前epoch进度：{:.2%}'.format((i+1)/len(dataloader)), end='')

            # 正向传播
            y_pre = model.forward(features)
            # 反向传播
            optimizer.zero_grad()
            loss = criterion(y_pre, labels)
            loss.backward()
            optimizer.step()

            # 统计参数
            # _, y_pre = torch.max(y_pre, 1)
            # y_pre = [0 if x < 0.5 else 1 for x in y_pre]
            acc.update(y_pre, labels)
            f1.update(y_pre, labels)
            recall.update(y_pre, labels)
            precision.update(y_pre, labels)

        acc_avg = acc.compute()
        f1_avg = f1.compute()
        recall_avg = recall.compute()
        precision_avg = precision.compute()

        print('\n* 第 {} 个epoch, acc: {}, f1: {}, precision: {}, recall: {}'.format(epoch+1, acc_avg, f1_avg, recall_avg, precision_avg))

    return model


def train_model_dl(model, train_set, feature_list, epochs=20):
    dataset = DatasetIoT(train_set, feature_list)
    model = train(model, dataset, epochs=epochs)
    return model


def predict_dl(model, test_set, feature_list):
    # 定义损失函数、优化器、DataLoader
    dataset = DatasetIoT(test_set, feature_list)
    dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

    # 开始训练
    acc = torchmetrics.Accuracy(task='multiclass', num_classes=2, average='micro')
    f1 = torchmetrics.F1Score(task='multiclass', num_classes=2, average='macro')
    recall = torchmetrics.Recall(task='multiclass', num_classes=2, average='macro')
    precision = torchmetrics.Precision(task='multiclass', num_classes=2, average='macro')

    for i, (labels, features) in enumerate(dataloader):
        print('\r开始预测：{:.2%}'.format((i + 1) / len(dataloader)), end='')

        # 正向传播
        y_pre = model.forward(features)

        # 统计参数
        # _, y_pre = torch.max(y_pre, 1)
        # y_pre = [0 if x < 0.5 else 1 for x in y_pre]
        acc.update(y_pre, labels)
        f1.update(y_pre, labels)
        recall.update(y_pre, labels)
        precision.update(y_pre, labels)

    acc_avg = acc.compute()
    f1_avg = f1.compute()
    recall_avg = recall.compute()
    precision_avg = precision.compute()
    print('\n* acc: {}, f1: {}, precision: {}, recall: {}'.format(acc_avg, f1_avg, recall_avg, precision_avg))

    return acc_avg, f1_avg, recall_avg, precision_avg



def evaluate_model_dl(model, train_set, test_set, feature_list, is_print=False):
    print('* evaluate : {}'.format(feature_list))
    model = train_model_dl(model, train_set, feature_list, epochs=5)

    redirect = Redirect()
    redirect.transform()
    evaluate_model_dl_for_time(model, train_set, test_set, feature_list, is_print=False)
    redirect.restore()
    line_time_list = redirect.content.split('\n')
    cal_time = 1
    for line in line_time_list:
        if 'predict_dl' in line:
            line_not_none = []
            for x in line.split(' '):
                if x != '':
                    line_not_none.append(x)
            cal_time = float(line_not_none[2]) / (10 ** 7)
    tracemalloc.start()
    acc, f1, precision, recall = predict_dl(model, test_set, feature_list)
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
    tracemalloc.stop()
    return f1, peak / 10 ** 6, cal_time, len(test_set) / cal_time


if __name__ == '__main__':
    rf = RandomForest()
    lr = LogisticR()
    svm = SVM()
    knn = KNN()
    lgb = LGB()
    dtree = Tree()
    model = dtree
    gru = GRU()
    mlp = MLP()

    cls_ml_list = [rf]
    cls_dl_list = [gru, mlp]

    dataset_name_list = ['iot-23-gate']
    for dataset_name in dataset_name_list:
        dst_res_path = './{}.csv'.format(dataset_name)

        feature_subset_dict = {
            'iot-23-gate': ['speed', 'bytes_mean'],
            'medbiot': ['bytes_mean', 'speed'],
            'ciciot2023': ['ip_uniq_ratio', 'ip_freq_var', 'time_interval_kur', 'speed'],
            'bot-iot': ['sport_uniq_ratio', 'dport_freq_mean', 'bytes_mean', 'speed']
        }
        class_num = {
            'iot-23-gate': 2,
            'medbiot': 2,
            'ciciot2023': 2,
            'bot-iot': 3
        }

        feature_subset = feature_subset_dict[dataset_name]

        df = pd.read_csv('../../dataset/chunk/{}/df_10000.csv'.format(dataset_name))
        df = pd.DataFrame(df, columns=feature_subset + ['label'])
        # df = sample_df(df)
        train_set, test_set = split_dataset(df)

        for cls in cls_ml_list:
            f1, peak, cal_time, sps = evaluate_model_ml(cls, train_set, test_set, feature_subset, is_print=True)
            print(f1, peak, cal_time, sps)
            df = pd.DataFrame({
                'dataset':[dataset_name],
                'model': [str(cls)],
                'f1': [f1],
                'memory(MB)': [peak],
                'time(s)': [cal_time],
                'SPS': [sps]
            })
            df.to_csv(dst_res_path, header=(not os.path.exists(dst_res_path)), index=False, mode='a')

        for cls in cls_dl_list:
            cls.set_input_output(len(feature_subset), class_num[dataset_name])
            f1, peak, cal_time, sps = evaluate_model_dl(cls, train_set, test_set, feature_subset, is_print=True)
            print(f1, peak, cal_time, sps)
            df = pd.DataFrame({
                'dataset': [dataset_name],
                'model': [str(cls)[:3]],
                'f1': [f1],
                'memory(MB)': [peak],
                'time(s)': [cal_time],
                'SPS': [sps]
            })
            df.to_csv(dst_res_path, header=(not os.path.exists(dst_res_path)), index=False, mode='a')

        # joblib.dump(cls, 'pkl/ad_ciciot2023.pkl', compress=3)

    # f1, peak, cal_time, sps = evaluate_model_dl(gru, train_set, eval_set, feature_list, is_print=True)
    # print(f1, peak, cal_time, sps)

