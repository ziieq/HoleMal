import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = None

    def set_input_output(self, n1, n2):
        self.mlp = nn.Sequential(
            nn.Linear(n1, 16),
            nn.Sigmoid(),
            nn.Linear(16, 8),
            nn.Sigmoid(),
            nn.Linear(8, n2),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = self.mlp(x)
        return res


if __name__ == '__main__':
    model = MLP()
    input = torch.rand(64, 5)   # batch * sentence_len * word_len
    output = model(input)
    print(output.size())            # torch.Size([64, 2])