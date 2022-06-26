# ライブラリのインポート
import torch, json, math
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
from argparse import ArgumentParser
from torch.optim import Adam, SGD, Adadelta

# 自作ライブラリのインポート
from DNN import linear_regression
from DNN import utils

log = utils.get_logger()

class MainModel(nn.Module):
    def __init__(self, args):
        super(MainModel, self).__init__()

        self.linear_regression = linear_regression.LinearRegression(args)

    def forward(self, data, target_tensor):

        y_hat = self.linear_regression(data["features_tensor"])
        y_hat = torch.squeeze(y_hat, 1)
        loss = torch.sqrt(F.mse_loss(y_hat, target_tensor))

        return loss, y_hat
