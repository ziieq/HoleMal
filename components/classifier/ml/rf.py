from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix


class RandomForest:
    def __init__(self):
        self.clf = RandomForestClassifier(max_leaf_nodes=120, n_estimators=3)

    def fit(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def show_feature_importance(self, features_list):
        df_dict = {}
        for i, col in enumerate(features_list):
            df_dict[col] = self.clf.feature_importances_[i]
        df_list = sorted(df_dict.items(), key=lambda x: x[1], reverse=True)
        print(df_list)
        return df_list

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
        FPR = min(FP / (FP + TN))

        return acc, f1, recall, precision, FPR

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




