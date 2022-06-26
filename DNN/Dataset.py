# ライブラリのインポート
import math
import random
import torch

# 自作ライブラリのインポート
from DNN import utils

log = utils.get_logger()

class Dataset:

    def __init__(self, samples, batch_size):
        self.samples = samples
        self.batch_size = batch_size
        self.num_batches = math.ceil(len(self.samples) / batch_size) # 各イテレーションのバッチ数

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        batch = self.raw_batch(index)
        return self.padding(batch)

    def raw_batch(self, index):
        assert index < self.num_batches, "batch_idx %d > %d" % (index, self.num_batches)
        batch = self.samples[index * self.batch_size: (index + 1) * self.batch_size]

        return batch

    def padding(self, samples):
        batch_size = len(samples)
        features_list = []
        target_list = []
        for i, s in enumerate(samples):
            features_list.append(s.features)
            target_list.append(s.target)

        data = {
            "features_tensor": torch.tensor(features_list).float(),
            "target_tensor": torch.tensor(target_list).float()
        }

        return data

    def shuffle(self):
        random.shuffle(self.samples)
