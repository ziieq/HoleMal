from sklearn.model_selection import train_test_split
import pandas as pd
from components.classifier.rf import RandomForest
from components.classifier.svm import SVM
from components.classifier.lr import LogisticR
from components.classifier.knn import KNN
from components.classifier.lgb import LGB
from components.classifier.tree import Tree
from sklearn.model_selection import KFold
import numpy as np
from line_profiler import LineProfiler
from functools import wraps
import tracemalloc
import joblib


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


@func_line_time
def cross_validation_ad(dataset, features, res_dict, model=None, save_path='./model.pkl', ):

    X = pd.read_csv(r'./dataset/df.csv'.format(dataset))
    X = pd.DataFrame(X, columns=features)

    X = X.sample(frac=1.0)

    # 构造二分类
    y = X.pop('label')

    KF = KFold(n_splits=10, shuffle=False)

    m_acc_list, m_f1_list, m_recall_list, m_precision_list = [], [], [], []
    for m in model:
        feature_score_dict = {}
        acc_list, f1_list, recall_list, precision_list = [], [], [], []
        for train_index, test_index in KF.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            # print(X_train.info())   # 222751
            # print(X_test.info())    # 24750
            m.fit(X_train.values, y_train)
            try:
                score_list = m.show_feature_importance(X_train.columns)
                for feature in score_list:
                    feature_score_dict.setdefault(feature[0], 0)
                    feature_score_dict[feature[0]] = feature[1]

            except Exception as e:
                pass

            """ 预测 """
            print(X_test.shape)
            tracemalloc.start()
            y_pre = m.predict(X_test.values)
            current, peak = tracemalloc.get_traced_memory()
            print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
            tracemalloc.stop()
            joblib.dump(m, save_path, compress=3)

            acc, f1, recall, precision = m.evaluate(y_test, y_pre)
            acc_list.append(acc)
            f1_list.append(f1)
            recall_list.append(recall)
            precision_list.append(precision)

        if feature_score_dict:
            feature_score_list = sorted(feature_score_dict.items(), key=lambda x: x[1], reverse=True)
            print('importance: {}'.format(feature_score_list))
            # draw_importance(feature_score_list)
        # print('len_mean: {}'.format(np.mean(len_mean)))
        print(m)
        print('acc:', np.mean(acc_list))
        print('f1:', np.mean(f1_list))
        print('recall:', np.mean(recall_list))
        print('precision:', np.mean(precision_list))
        m_acc_list.append(np.mean(acc_list))
        m_f1_list.append(np.mean(f1_list))
        m_recall_list.append(np.mean(recall_list))
        m_precision_list.append(np.mean(precision_list))

    print('All acc:', np.mean(m_acc_list))
    print('All f1:', np.mean(m_f1_list))
    print('All recall:', np.mean(m_recall_list))
    print('All precision:', np.mean(m_precision_list))
    res_dict['acc'].append(np.mean(m_acc_list))
    res_dict['f1'].append(np.mean(m_f1_list))
    res_dict['recall'].append(np.mean(m_recall_list))
    res_dict['precision'].append(np.mean(m_precision_list))


@func_line_time
def cross_validation_mfd(dataset, features_mfd_rfe_rf_final, res_dict, model=None, save_path='./model.pkl'):

    X = pd.read_csv(r'./dataset/df.csv'.format(dataset))
    X = pd.DataFrame(X, columns=features_mfd_rfe_rf_final)

    X_mal = X[X['label'] == 1]
    X_mal.pop('label')

    res = X_mal['family'].value_counts()
    print(res)

    X_sample_list = []
    sample_dict = {
        'mirai': 8000,
        'gafgyt': 8000,
        'ircbot': 8000,
        'kenjiro': 8000,
        'okiru': 2000,
        'muhstik': 200,
        'hideandseek': 200,
        'trojan': 200,
        'troii': 200,
        'hajime': 200,
        'hakai': 200,
    }
    for f in set(list(X_mal['family'])):
        X_sample_list.append(X_mal[X_mal['family'] == f].sample(n=sample_dict[f], replace=True))

    X_mal = pd.concat(X_sample_list, axis=0)
    X_mal = X_mal.sample(frac=1.0)

    # 构造家族分类
    res = X_mal['family'].value_counts()
    # print(res)
    family = X_mal.pop('family')
    family_num_dict = {}
    num = 0
    for fa in set(list(family)):
        family_num_dict[fa] = num
        num += 1
    y_family = family.apply(lambda x:family_num_dict[x])

    KF = KFold(n_splits=10, shuffle=False)

    m_acc_list, m_f1_list, m_recall_list, m_precision_list = [], [], [], []
    for m in model:
        feature_score_dict = {}
        acc_list, f1_list, recall_list, precision_list = [], [], [], []
        for train_index, test_index in KF.split(X_mal):
            X_train, X_test = X_mal.iloc[train_index], X_mal.iloc[test_index]
            y_train, y_test = y_family.iloc[train_index], y_family.iloc[test_index]
            m.fit(X_train.values, y_train)
            try:
                score_list = m.show_feature_importance(X_train.columns)
                for feature in score_list:
                    feature_score_dict.setdefault(feature[0], 0)
                    feature_score_dict[feature[0]] += feature[1]
            except Exception as e:
                pass

            """ 预测 """
            tracemalloc.start()
            print(X_test.shape)
            y_pre = m.predict(X_test.values)
            current, peak = tracemalloc.get_traced_memory()
            print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
            tracemalloc.stop()
            joblib.dump(m, save_path, compress=3)

            # len_mean.append(np.mean(list(X_test['pkt_len_mean'])))
            # t = classification_report(y_test, y_pre, target_names=set(list([str(x) for x in y_pre])))
            # print(t)
            acc, f1, recall, precision = m.evaluate(y_test, y_pre)
            acc_list.append(acc)
            f1_list.append(f1)
            recall_list.append(recall)
            precision_list.append(precision)

        if feature_score_dict:
            for key in feature_score_dict.keys():
                feature_score_dict[key] /= 10
            feature_score_list = sorted(feature_score_dict.items(), key=lambda x: x[1], reverse=True)
            print('final importance: {}'.format(feature_score_list))
            # draw_importance(feature_score_list)

        # print('len_mean: {}'.format(np.mean(len_mean)))
        print(m)
        print('acc:', np.mean(acc_list))
        print('f1:', np.mean(f1_list))
        print('recall:', np.mean(recall_list))
        print('precision:', np.mean(precision_list))
        m_acc_list.append(np.mean(acc_list))
        m_f1_list.append(np.mean(f1_list))
        m_recall_list.append(np.mean(recall_list))
        m_precision_list.append(np.mean(precision_list))
    print('All acc:', np.mean(m_acc_list))
    print('All f1:', np.mean(m_f1_list))
    print('All recall:', np.mean(m_recall_list))
    print('All precision:', np.mean(m_precision_list))
    res_dict['acc'].append(np.mean(m_acc_list))
    res_dict['f1'].append(np.mean(m_f1_list))
    res_dict['recall'].append(np.mean(m_recall_list))
    res_dict['precision'].append(np.mean(m_precision_list))


if __name__ == '__main__':
    rf = RandomForest()
    lr = LogisticR()
    svm = SVM()
    knn = KNN()
    lgb = LGB()
    dtree = Tree()
    model = dtree

    new_f = ['bytes_mean', 'ip_uniq_ratio',  'speed', 'dport_uniq_ratio']
    res_dict = {'acc': [], 'f1': [], 'recall': [], 'precision': []}
    cross_validation_ad(new_f + ['label'], res_dict, model=[rf], save_path='pkl/ad.pkl')
    print(res_dict)

    new_f = ['bytes_mean', 'ssh', 'speed', 'http', 'sport_uniq_ratio']
    res_dict = {'acc': [], 'f1': [], 'recall': [], 'precision': []}
    cross_validation_mfd(new_f + ['family', 'label'], res_dict, model=[rf], save_path='pkl/mfd.pkl')
    print(res_dict)
