import numpy as np
import torch
import torch.utils.data as data
import pytorch_lightning as pl

from mylib.config import Configurable
from mylib.config import save
from mylib.utils import get_mf

class RouteDatasetWithZeros(data.Dataset, Configurable):
    n_of_route: save
    is_2ax: save
    add_noise: save
    divide: save

    def __init__(self, n_of_route, is_2ax=False, add_noise=False, divide=1) -> None:
        self.n_of_route = n_of_route
        self.is_2ax = is_2ax
        self.add_noise = add_noise
        self.divide = divide

    def set_route(self, df, route_gen):
        self.df = df
        origin_route = route_gen.get_route_list(self.n_of_route)
        divided_route = []  # ルートを分割
        for route in origin_route:
            X = route[0]
            Y = route[1]
            step = len(X) // self.divide
            for i in range(self.divide):
                divided_route.append([X[i*step:(i+1)*step], Y[i*step:(i+1)*step]])
            
        self.route = divided_route

    def __len__(self) -> int:
        return len(self.route)

    # -> tuple[torch.Tensor, torch.Tensor]
    def __getitem__(self, idx):
        # print("getitem")
        X = self.route[idx][0]
        Y = self.route[idx][1]

        MF = get_mf(self.df, X, Y, self.is_2ax, self.add_noise)
        X = [float(x * 0.1) for x in X]
        Y = [float(y * 0.1) for y in Y]
        XY = np.column_stack((X, Y))

        zero = [0 for i in range(len(X))]
        
        ZEROS = np.column_stack((zero, zero))


        mf_and_d = torch.cat([torch.Tensor(MF), torch.Tensor(ZEROS)], dim=1)
        
        return mf_and_d, torch.Tensor(XY)