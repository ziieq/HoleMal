import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=16,  num_layers=1, batch_first=True, bidirectional=True)
        self.mlp = None

    def set_input_output(self, n1, n2):
        self.mlp = nn.Sequential(
            nn.Linear(32, 16),
            nn.Sigmoid(),
            nn.Linear(16, n2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, hidden = self.gru(x.to(torch.float32))
        x = out[:, -1, :]
        res = self.mlp(x)
        return res


if __name__ == '__main__':
    model = GRU()
    input = torch.rand(64, 6, 1)   # batch * sentence_len * word_len
    output = model(input)
    print(output.size())            # torch.Size([64, 2])