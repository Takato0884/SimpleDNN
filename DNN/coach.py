# ライブラリのインポート
import copy
import time
import numpy as np
import torch
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import mean_squared_error

# 自作ライブラリのインポート
from DNN import utils

log = utils.get_logger()

class Coach:

    def __init__(self, trainset, valset, testset, model, opt, args):
        self.trainset = trainset
        self.valset = valset
        self.testset = testset
        self.model = model
        self.opt = opt
        self.args = args

    def train(self):

        best_valid_rmse, best_epoch_rmse, best_state = None, None, None
        train_rmse_list, valid_rmse_list, test_rmse_list = [], [], []
        train_loss_list, valid_loss_list, test_loss_list = [], [], []
        valid_golds_list, valid_preds_list = [], []
        test_golds_list, test_preds_list = [], []

        # Train
        for epoch in range(1, self.args.epochs + 1):
            train_rmse, train_loss = self.train_epoch(epoch)

            valid_rmse, valid_loss, valid_golds, valid_preds = self.evaluate()
            log.info("[Val set] [rmse {:.4f}]".format(valid_rmse))
            if best_valid_rmse is None or best_valid_rmse > valid_rmse:
                best_valid_rmse = valid_rmse
                log.info("best rmse model.")

            test_rmse, test_loss, test_golds, test_preds = self.evaluate(test=True)
            log.info("[Test set] [rmse {:.4f}]".format(test_rmse))

            train_rmse_list.append(train_rmse)
            valid_rmse_list.append(valid_rmse)
            test_rmse_list.append(test_rmse)
            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)
            test_loss_list.append(test_loss)
            valid_golds_list.append(valid_golds)
            valid_preds_list.append(valid_preds)
            test_golds_list.append(test_golds)
            test_preds_list.append(test_preds)

        return {"valid_golds_list": valid_golds_list, "test_golds_list": test_golds_list, "valid_preds_list": valid_preds_list, "test_preds_list": test_preds_list,\
        "train_rmse_list": train_rmse_list, "valid_rmse_list": valid_rmse_list,"test_rmse_list": test_rmse_list, \
        "train_loss_list": train_loss_list, "valid_loss_list": valid_loss_list,"test_loss_list": test_loss_list, "best_valid_rmse": best_valid_rmse}

    def train_epoch(self, epoch):
        start_time = time.time()
        epoch_loss = 0
        self.model.train()

        # ここからバッチごとの処理
        golds = []
        preds = []
        for idx in tqdm(range(len(self.trainset)), desc="train epoch {}".format(epoch)):
            data = self.trainset[idx]
            golds.append(data["target_tensor"])
            self.model.zero_grad()
            for k, v in data.items():
                data[k] = v.to(self.args.device)
            nll, y_hat = self.model(data, data["target_tensor"])
            epoch_loss += nll.item()
            preds.append(y_hat.detach().to("cpu"))
            nll.backward()
            self.opt.step()

        golds = torch.cat(golds, dim=-1).numpy()
        preds = torch.cat(preds, dim=-1).numpy()
        rmse = np.sqrt(mean_squared_error(preds, golds))
        end_time = time.time()
        log.info("")
        log.info("[Epoch %d] [Loss: %f] [Time: %f]" %
                 (epoch, epoch_loss, end_time - start_time))

        return rmse, epoch_loss

    def evaluate(self, test=False):
        dataset = self.testset if test else self.valset
        epoch_loss = 0
        self.model.eval()
        with torch.no_grad():
            golds = []
            preds = []
            for idx in tqdm(range(len(dataset)), desc="test" if test else "dev"):
                data = dataset[idx]
                golds.append(data["target_tensor"])
                for k, v in data.items():
                    data[k] = v.to(self.args.device)
                nll, y_hat = self.model(data, data["target_tensor"])
                preds.append(y_hat.detach().to("cpu"))

            golds = torch.cat(golds, dim=-1).numpy()
            preds = torch.cat(preds, dim=-1).numpy()
            rmse = np.sqrt(mean_squared_error(preds, golds))

        return rmse, epoch_loss, golds, preds
