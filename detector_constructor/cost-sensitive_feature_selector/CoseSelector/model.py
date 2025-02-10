import torch.nn as nn
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import tree
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.metrics import confusion_matrix


class GRU(nn.Module):
    def __init__(self, device='cpu'):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=32,  num_layers=1, batch_first=True, bidirectional=True).to(device)
        self.mlp = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, x):
        out, (hidden, cell) = self.gru(x.to(torch.float32))
        out = out[:, -1, :]
        out = self.mlp(out)
        return out


class RandomForest(object):
    def __init__(self):
        self.clf = RandomForestClassifier(max_leaf_nodes=100, n_estimators=3)
        #self.clf = RandomForestClassifier(max_depth=5, n_estimators=10)
        #self.clf = svm.SVC(kernel='rbf',  max_iter=2)
        #self.clf = tree.DecisionTreeClassifier(max_leaf_nodes=10)
        # self.clf = GaussianNB()

    def fit(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def get_feature_importance(self, features_list):
        df_dict = {}
        for i, col in enumerate(features_list):
            df_dict[col] = self.clf.feature_importances_[i]
        df_list = sorted(df_dict.items(), key=lambda x: x[1], reverse=True)
        return df_list

    def predict(self, x_test):
        y_pre = self.clf.predict(x_test)
        return y_pre

    def evaluate(self, y_test, y_pre, print_res=False):
        acc = accuracy_score(y_test, y_pre)
        f1 = f1_score(y_test, y_pre, average='macro')
        recall = recall_score(y_test, y_pre, average='macro')
        precision = precision_score(y_test, y_pre, average='macro')
        auc = 0
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

        try:
            auc = roc_auc_score(y_test, y_pre, average='macro', multi_class='ovr')
        except Exception as e:
            pass
        if print_res:
            print()
            print('rf evaluation results:')
            print('acc: ', acc)
            print('f1: ', f1)
            print('recall: ', recall)
            print('precision: ', precision)
            print('auc:', auc)
            print()
        return acc, f1, recall, precision, auc, FPR


class Transformer(nn.Module):

    def __init__(self, dim, device='cpu'):
        super(Transformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=dim).to(device)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1).to(device)

        self.mlp = nn.Sequential(
            nn.Linear(dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        ).to(device)

    def forward(self, x):
        out = self.encoder(x)
        out = self.mlp(out)
        return out


class MLP(nn.Module):

    def __init__(self, input_dim, out_dim, device='cpu'):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Sigmoid(),
            nn.Linear(16, out_dim),
            nn.Sigmoid()
        ).to(device)

    def forward(self, x):
        out = self.mlp(x)
        return out

if __name__ == '__main__':
    # model = Transformer(63, device='cuda:0')
    model = MLP(63, 2, device='cuda:0')
    input = torch.rand(100, 63).to('cuda:0')
    output = model(input)
    print(output)
    print(output.size())