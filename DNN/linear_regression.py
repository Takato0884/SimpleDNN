# ライブラリのインポート
import torch, json, math
import torch.nn as nn
from torch.nn import functional as F

# 自作ライブラリのインポート
from DNN import utils

log = utils.get_logger()

class LinearRegression(nn.Module):
    def __init__(self, args):
        super(LinearRegression, self).__init__()

        self.drop = nn.Dropout(args.drop_rate)
        self.lin1 = nn.Linear(args.input_size, args.mid_size1)
        self.lin2 = nn.Linear(args.mid_size1, args.mid_size2)
        self.lin3 = nn.Linear(args.mid_size2, 1)

    def forward(self, h):

        hidden1 = self.drop(F.relu(self.lin1(h)))
        hidden2 = self.drop(F.relu(self.lin2(hidden1)))
        scores = self.lin3(hidden2)

        return scores
