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

# dfの内容を配列にキャッシュ
df_array = None

def set_df_array(df):
    global df_array

    # 最大値を取得して2次元リストを初期化
    max_x = df["x"].max()
    max_y = df["y"].max()

    df_array = [[None for _ in range(max_y + 1)] for _ in range(max_x + 1)]

    # 2次元リストにデータを格納
    for _, row in df.iterrows():
        x = int(row["x"])  # 整数に変換
        y = int(row["y"])
        df_array[x][y] = (row["MF_X"], row["MF_Y"], row["MF_Z"])

def get_mf(df, X, Y, is_2ax=False, add_noise=False):
    MF = []

    if df_array is None:
        print("Cache not found. Set cache.")
        set_df_array(df)
    
    for x, y in zip(X, Y):
        try:
            # MF.append(df[(df["x"] == x) & (df["y"] == y)]["MF"].values[0])
            mf_x, mf_y, mf_z = df_array[x][y]

            if add_noise:
                mean = 0.0  # 平均
                std = 0.381  # 標準偏差
                mf_x += np.random.normal(mean, std)
                mf_y += np.random.normal(mean, std)
                mf_z += np.random.normal(mean, std)

            if is_2ax:
                horizontal = np.sqrt(mf_x**2 + mf_y**2)
                virtical = mf_z        
                MF.append([horizontal, virtical])
            else:
                MF.append([mf_x, mf_y, mf_z])

        except Exception as e:  # 例外の詳細を取得
            print(f"Error at ({x}, {y}): {e}")
            MF.append(-1)
    return MF