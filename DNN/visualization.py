# ライブラリのインポート
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd

class Visualization:
    def create_overall_dataframe(ret):
        overall_ret_dict = {"train_rmse_list": [], "valid_rmse_list": [], "test_rmse_list": []}

        for k in ["train_rmse_list", "valid_rmse_list","test_rmse_list"]:
            rmse_list = ret[k]
            sum_rmse = sum(rmse_list)
            for i in range(len(rmse_list)):
                rmse_list[i] = (rmse_list[i] / sum_rmse)
            overall_ret_dict[k] = rmse_list

        df = pd.DataFrame(overall_ret_dict["train_rmse_list"], columns = ['train_rmse']).reset_index().rename(columns={'index': 'epoch'})
        df["valid_rmse"] = overall_ret_dict["valid_rmse_list"]
        df["test_rmse"] = overall_ret_dict["test_rmse_list"]
        df["epoch"] += 1

        return df

    def show_curve(df):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
        go.Scatter(x=df["epoch"], y=df['train_rmse'], name="train (RMSE)"),
        secondary_y=False)
        fig.add_trace(
        go.Scatter(x=df["epoch"], y=df['valid_rmse'], name="valid (RMSE)"),
        secondary_y=False)
        fig.add_trace(
        go.Scatter(x=df["epoch"], y=df['test_rmse'], name="test (RMSE)"),
        secondary_y=False)
        fig.update_layout(xaxis=dict(title="epoch", rangemode='tozero'))
        fig.update_layout(yaxis=dict(title='RMSE (relative freq)'))
        fig.update_yaxes(range=(0.015, 0.008), secondary_y=False)
        fig.update_layout(plot_bgcolor="white")
        fig.update_xaxes(linecolor='black',mirror=True)
        fig.update_yaxes(linecolor='black',mirror=True)
        fig.show()
