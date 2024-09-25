import numpy as np
import torch
import torch.utils.data as data
import pytorch_lightning as pl

from mylib.config import Configurable
from mylib.config import save

class RouteDatasetWithRouteDiff(data.Dataset, Configurable):
    n_of_route: save
    diff_size: save

    def __init__(self, n_of_route, diff_size) -> None:
        self.n_of_route = n_of_route
        self.diff_size = diff_size

    def set_route(self, df, route_gen):
        self.df = df
        self.route = route_gen.get_route_list(self.n_of_route)

    def __len__(self) -> int:
        return len(self.route)

    # -> tuple[torch.Tensor, torch.Tensor]
    def __getitem__(self, idx):
        # print("getitem")
        X = self.route[idx][0]
        Y = self.route[idx][1]

        MF = getMF(self.df, X, Y)
        X = [float(x * 0.1) for x in X]
        Y = [float(y * 0.1) for y in Y]
        XY = np.column_stack((X, Y))
        X = [X[i] - X[0] for i in range(len(X))]
        Y = [Y[i] - Y[0] for i in range(len(Y))]

        # 適当な誤差を生成
        diff_X = [np.random.normal(-self.diff_size, self.diff_size) for _ in X]
        diff_Y = [np.random.normal(-self.diff_size, self.diff_size) for _ in Y]
        diff_X[0] = 0
        diff_Y[0] = 0
        # 誤差を累積
        X = [X[i] + sum(diff_X[:i]) for i in range(len(X))]
        Y = [Y[i] + sum(diff_Y[:i]) for i in range(len(Y))]

        # 原点中心に回転
        theta = np.arctan2(Y[1], X[1])
        rotation_angle = np.pi / 2 - theta
        X_rotated = [X[i] * np.cos(rotation_angle) - Y[i] * np.sin(rotation_angle) for i in range(len(X))]
        Y_rotated = [X[i] * np.sin(rotation_angle) + Y[i] * np.cos(rotation_angle) for i in range(len(Y))]

        # # 最初の移動の大きさを1にする
        norm = 1 / Y_rotated[1]
        X_rotated = [x * norm for x in X_rotated]
        Y_rotated = [y * norm for y in Y_rotated]

        XY_zero = np.column_stack((X_rotated, Y_rotated))

        mf_and_d = torch.cat([torch.Tensor(MF), torch.Tensor(XY_zero)], dim=1)
        
        return mf_and_d, torch.Tensor(XY)
    

def getMF(df, X, Y):
    MF = []
    for x, y in zip(X, Y):
        try:
            # MF.append(df[(df["x"] == x) & (df["y"] == y)]["MF"].values[0])
            d = df[(df["x"] == x) & (df["y"] == y)]
            MF.append([d["MF_X"].values[0], d["MF_Y"].values[0], d["MF_Z"].values[0]])
        except:
            print(f"Error: {x}, {y}")
            MF.append(-1)
    return MF

