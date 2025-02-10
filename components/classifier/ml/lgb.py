from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import lightgbm as lgb


class LGB(object):

    def __init__(self):
        self.clf = lgb.LGBMClassifier(num_leaves=60, n_estimators=6)

    def fit(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

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
        return acc, f1, recall, precision, -1

    def show_feature_importance(self, features_list):
        df_dict = {}
        for i, col in enumerate(features_list):
            df_dict[col] = self.clf.feature_importances_[i]
        df_list = sorted(df_dict.items(), key=lambda x: x[1], reverse=True)
        print(df_list)
        return df_list

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

        for item in sorted(ip2score_dict.items(), key=lambda x: x[1], reverse=True):
            if item[1] >= 0.5:
                print('malicious ip: ', item[0])
            else:
                print('benign ip: ', item[0])
