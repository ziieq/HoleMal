from torch.utils.data import Dataset
import pandas as pd
import torch


class DatasetIoT(Dataset):
    def __init__(self, df, feature_list):
        """
        读取数据集csv文件
        """
        self.feature_list = feature_list
        self.df = pd.DataFrame(df, columns=feature_list + ['label'])

    def __getitem__(self, idx):
        """
        返回第idx条数据。通常将label与features分开返回。
        :param idx: 数据编号
        :return: 第idx条数据
        """
        data = self.df.iloc[idx]
        label = torch.tensor(data.pop('label'), dtype=torch.long)
        data = torch.Tensor(data)
        return label, data

    def __len__(self):
        """
        返回数据集中的数据总条数
        :return: 数据总条数
        """
        return self.df.shape[0]


if __name__ == '__main__':
    feature_list = ['ip_uniq_ratio', 'dport_uniq_ratio', 'bytes_mean', 'speed', 'http']
    train_set = pd.read_csv(r'E:\Main\Engineering\PyCharm\Project\IoT\HoleMal(开源)\HoleMal\dataset\chunk\bot-iot\df_10000.csv')
    dataset = DatasetIoT(train_set, feature_list)
    print(dataset.__getitem__(8))   # (True, tensor([[172.4700], [103.9200], [103.6700], [4.0000], [3.2500], [112.8500]], dtype=torch.float64))
    print(dataset.__len__())        # 1500
