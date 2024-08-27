import numpy as np
import torch
import torch.utils.data as data
import pytorch_lightning as pl

from mylib.config import Configurable
from mylib.config import save

class MyDataModule(pl.LightningDataModule, Configurable):
    n_of_route: save
    batch_size: save

    def __init__(self, n_of_route, batch_size, route_gen, df):
        super().__init__()
        self.n_of_route = n_of_route
        self.batch_size = batch_size

        test_size = int(n_of_route * 0.2)
        train_size = int((n_of_route - test_size) * 0.8)
        val_size = n_of_route - train_size - test_size

        self.train_dataset = RouteDataset(df, train_size, route_gen)
        self.val_dataset = RouteDataset(df, val_size, route_gen)
        self.test_dataset = RouteDataset(df, test_size, route_gen)
        

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=7, persistent_workers=True)

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=7, persistent_workers=True)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=7, persistent_workers=True)


class MyDataModuleWithRoute(pl.LightningDataModule, Configurable):
    n_of_route: save
    batch_size: save

    def __init__(self, n_of_route, batch_size, route_gen, df):
        super().__init__()
        self.n_of_route = n_of_route
        self.batch_size = batch_size

        test_size = int(n_of_route * 0.2)
        train_size = int((n_of_route - test_size) * 0.8)
        val_size = n_of_route - train_size - test_size

        self.train_dataset = RouteDatasetWithRoute(df, train_size, route_gen)
        self.val_dataset = RouteDatasetWithRoute(df, val_size, route_gen)
        self.test_dataset = RouteDatasetWithRoute(df, test_size, route_gen)
        

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=7, persistent_workers=True)

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=7, persistent_workers=True)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=7, persistent_workers=True)



class RouteDataset(data.Dataset):
    def __init__(self, df, n_of_route, route_gen) -> None:
        self.df = df
        self.size = n_of_route
        self.route_gen = route_gen
        self.route = route_gen.get_route_list(n_of_route)

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

        # distance = [np.sqrt((X[i] - X[i + 1])**2 + (Y[i] - Y[i + 1])**2) for i in range(len(X) - 1)]        
        # distance.insert(0, 0)

        distance = [0 for i in range(len(X))]    

        mf_and_d = torch.cat([torch.Tensor(MF), torch.Tensor(distance).unsqueeze(1)], dim=1)
        
        return mf_and_d, torch.Tensor(XY)
    

class RouteDatasetWithRoute(data.Dataset):
    def __init__(self, df, n_of_route, route_gen) -> None:
        self.df = df
        self.size = n_of_route
        self.route_gen = route_gen
        self.route = route_gen.get_route_list(n_of_route)

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
        diff_X = [np.random.normal(0, 0.2) for _ in X]
        diff_Y = [np.random.normal(0, 0.2) for _ in Y]
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
        # print(f"X_rotated: {X_rotated}")
        # print(f"Y_rotated: {Y_rotated}")

        # X_rotated = [0 for i in range(len(X))]
        # Y_rotated = [0 for i in range(len(Y))]

        

        # theta = np.arctan(rot_Y[1]/ rot_X[1])
        # print(f"theta: {theta}")


        # XY_zero = np.column_stack((X, Y))
        XY_zero = np.column_stack((X_rotated, Y_rotated))

        distance = [0 for i in range(len(X))]    

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

