import numpy as np
import torch
import torch.utils.data as data
import pytorch_lightning as pl

from mylib.config import Configurable
from mylib.config import save

class RouteDatasetWithDistance(data.Dataset, Configurable):
    n_of_route: save

    def __init__(self, n_of_route) -> None:
        self.n_of_route = n_of_route

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


        # 点間の距離
        distance = [np.sqrt((X[i] - X[i + 1])**2 + (Y[i] - Y[i + 1])**2) for i in range(len(X) - 1)]      
        distance.insert(0, 0)

        # 初速が不明のため、最初の移動の大きさを1にする
        norm = 1 / distance[1]
        distance = [d * norm for d in distance]

        zero = [0 for i in range(len(X))]

        DISTANCE = np.column_stack((distance, zero)) 

        mf_and_d = torch.cat([torch.Tensor(MF), torch.Tensor(DISTANCE)], dim=1)
        
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
