import random
import torch
import numpy as np
import datetime

def fix_seeds(seed=0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_datetime() -> str:
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    return now.strftime('%Y-%m-%d-%H-%M')

def get_mf(df, X, Y, is_2ax=False, add_noise=False):
    MF = []
    for x, y in zip(X, Y):
        try:
            # MF.append(df[(df["x"] == x) & (df["y"] == y)]["MF"].values[0])
            d = df[(df["x"] == x) & (df["y"] == y)]
            x = d["MF_X"].values[0] 
            y = d["MF_Y"].values[0]
            z = d["MF_Z"].values[0]

            if add_noise:
                mean = 0.0  # 平均
                std = 0.381  # 標準偏差
                x += np.random.normal(mean, std)
                y += np.random.normal(mean, std)
                z += np.random.normal(mean, std)

            if is_2ax:
                horizontal = np.sqrt(x**2 + y**2)
                virtical = z        
                MF.append([horizontal, virtical])
            else:
                MF.append([x, y, z])

        except:
            print(f"Error: {x}, {y}")
            MF.append(-1)
    return MF