from sklearn import svm
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


class SVM(object):
    def __init__(self):
        self.clf = svm.SVC(max_iter=50000)

    def fit(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    # def show_feature_importance(self, features_list):
    #     df_dict = {}
    #     for i, col in enumerate(features_list):
    #         df_dict[col] = self.clf.feature_importances_[i]
    #     df_list = sorted(df_dict.items(), key=lambda x: x[1], reverse=True)
    #     print(df_list)

    def predict(self, x_test):
        y_pre = self.clf.predict(x_test)
        return y_pre

    def evaluate(self, y_test, y_pre, print_res=False):
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

    def check_ip(self, y_pre, ip_test):
        ip2score_dict = {}
        for i in range(len(y_pre)):
            if y_pre[i] == 1:
                ip2score_dict.setdefault(ip_test.iloc[i], [0, 0])[1] += 1
            else:
                ip2score_dict.setdefault(ip_test.iloc[i], [0, 0])[0] += 1
        for ip in ip2score_dict.keys():
            if ip2score_dict[ip][0] == 0:
                ip2score_dict[ip] = 1
            else:
                ip2score_dict[ip] = ip2score_dict[ip][1] / (ip2score_dict[ip][0] + ip2score_dict[ip][1])

        for item in sorted(ip2score_dict.items(), key=lambda x:x[1], reverse=True):
            if item[1] >= 0.5:
                print('malicious ip: ', item[0])
            else:
                print('benign ip: ', item[0])




