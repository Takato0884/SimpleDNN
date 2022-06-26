import sys
import random
import logging

import numpy as np
import torch
import pickle


def set_seed(seed):
    """Sets random seed everywhere."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

# Loggingの設定
# https://docs.python.org/ja/3/howto/logging.html#logging-basic-tutorial
def get_logger(level=logging.INFO):
    log = logging.getLogger(__name__) # ロガーのインスタンス作成
    if log.handlers:
        return log
    log.setLevel(level) # 適切な出力先に振り分けられるべき最も低い深刻度を指定
    ch = logging.StreamHandler(sys.stdout) # ハンドラのインスタンス作成
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    ch.setFormatter(formatter) # ハンドラが使用するFormatterオブジェクトを選択
    log.addHandler(ch)
    return log


def save_pkl(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f)
