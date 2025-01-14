import numpy as np
import torch
import torch.utils.data as data
import pytorch_lightning as pl

from mylib.config import Configurable
from mylib.config import save
from mylib.utils import get_mf

class RouteDatasetWithRoute(data.Dataset, Configurable):
    n_of_route: save
    is_2ax: save
    add_noise: save
    divide: save
    comp: save

    def __init__(self, n_of_route, is_2ax=False, add_noise=False, divide=1, comp=-1) -> None:
        self.n_of_route = n_of_route
        self.is_2ax = is_2ax
        self.add_noise = add_noise
        self.divide = divide
        self.comp = comp

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
        # self.route = route_gen.get_route_list(self.n_of_route)


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
        X = [X[i] - X[0] for i in range(len(X))]
        Y = [Y[i] - Y[0] for i in range(len(Y))]

        # # 適当な誤差を生成
        # diff_X = [np.random.normal(0, 0.2) for _ in X]
        # diff_Y = [np.random.normal(0, 0.2) for _ in Y]
        # diff_X[0] = 0
        # diff_Y[0] = 0
        # # 誤差を累積
        # X = [X[i] + sum(diff_X[:i]) for i in range(len(X))]
        # Y = [Y[i] + sum(diff_Y[:i]) for i in range(len(Y))]

        # 原点中心に回転
        theta = np.arctan2(Y[1], X[1])
        rotation_angle = np.pi / 2 - theta
        X_rotated = [X[i] * np.cos(rotation_angle) - Y[i] * np.sin(rotation_angle) for i in range(len(X))]
        Y_rotated = [X[i] * np.sin(rotation_angle) + Y[i] * np.cos(rotation_angle) for i in range(len(Y))]

        # # 最初の移動の大きさを1にする
        norm = 1 / Y_rotated[1]
        X_rotated = [x * norm for x in X_rotated]
        Y_rotated = [y * norm for y in Y_rotated]

        length = len(X_rotated)
        

        XY_zero = np.column_stack((X_rotated, Y_rotated))

        mf_and_d = torch.cat([torch.Tensor(MF), torch.Tensor(XY_zero)], dim=1)
        XY = torch.Tensor(XY)

        comp_diff = self.comp - length
        if comp_diff > 0:
            mf_and_d = torch.cat([mf_and_d, torch.zeros(comp_diff, 4)], dim=0)
            XY = torch.cat([XY, torch.zeros(comp_diff, 2)], dim=0)


        # print("mf_and_d", mf_and_d)
        # print("XY", XY)


        dummy_n = np.random.randint(0, 10)

        dummy_mf_and_d = [[0, 0, 0, 0] for _ in range(length)]
        dummy_mf_and_d[0] = [dummy_n, dummy_n, dummy_n, dummy_n]
        dummy_mf_and_d = torch.Tensor(dummy_mf_and_d)

        dummy_XY = [[dummy_n, dummy_n] for _ in range(length)]
        dummy_XY = torch.Tensor(dummy_XY)

        # print("dummy_mf_and_d", dummy_mf_and_d)
        # print("dummy_XY", dummy_XY)

        return dummy_mf_and_d, dummy_XY
        
        return mf_and_d, XY


