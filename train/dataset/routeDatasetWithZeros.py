import numpy as np
import torch
import torch.utils.data as data
import pytorch_lightning as pl

from mylib.config import Configurable
from mylib.config import save

class RouteDatasetWithZeros(data.Dataset, Configurable):
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

        zero = [0 for i in range(len(X))]
        # zero = [0 for i in range(len(Y))]

        
        ZEROS = np.column_stack((zero, zero))


        mf_and_d = torch.cat([torch.Tensor(MF), torch.Tensor(ZEROS)], dim=1)
        
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
