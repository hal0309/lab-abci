import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

from mylib.config import Configurable
from mylib.config import save

class TransformerWithRouteDecoderAR(pl.LightningModule, Configurable):
    input_size: save
    d_model: save
    nhead: save
    num_layers: save
    out_length: save
    opt_lr: save

    def __init__(self, input_size=80, d_model=80, nhead=8, num_layers=2, out_length=40, opt_lr=1e-3):  
        super().__init__()
        self.save_hyperparameters()

        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.out_length = out_length
        self.opt_lr = opt_lr        

        # エンコーダ
        self.input_linear = nn.Linear(input_size, d_model)
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # デコーダ
        self.target_embed = nn.Linear(out_length, d_model)
        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, out_length)

    def forward(self, src, tgt):
        """
        Forward関数
        src: [batch_size, src_seq_len, input_size] -> Encoder入力
        tgt: [batch_size, tgt_seq_len, input_size] -> Decoder入力
        """
        # Encoderへの入力処理
        src = self.input_linear(src)  # [batch_size, src_seq_len, d_model]
        memory = self.transformer_encoder(src)  # [batch_size, src_seq_len, d_model]

        # Decoderへの入力処理
        tgt = self.target_embed(tgt)  # [batch_size, tgt_seq_len, d_model]
        output = self.transformer_decoder(tgt, memory)  # [batch_size, tgt_seq_len, d_model]

        # 出力層
        output = self.fc(output)  # [batch_size, tgt_seq_len, 2]
        return output
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.opt_lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.view(64, -1), y.view(64, -1)  # 入力と出力を整形

        batch_size, seq_len = y.shape
        shifted_y = torch.zeros_like(y)  # 初期化（全ゼロ）
        y_hat_list = []  # 逐次予測を格納するリスト

        # 自己回帰による逐次学習プロセス
        for t in range(0, seq_len, 2):
            # 現在のタイムステップで予測
            y_hat_step = self(x, shifted_y)

            # 予測結果をリストに追加
            y_hat_list.append(y_hat_step[:, t:t+2])

            # shifted_y を更新 (新しいテンソルに割り当て)
            if t + 2 < seq_len:
                shifted_y = shifted_y.clone()  # 新しいテンソルを作成
                shifted_y[:, t + 2] = y_hat_step[:, t]
                if t + 3 < seq_len:
                    shifted_y[:, t + 3] = y_hat_step[:, t + 1]

        # 逐次予測を結合して1つのテンソルに
        y_hat = torch.cat(y_hat_list, dim=1)

        # 損失を計算
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.view(64, -1), y.view(64, -1)

        batch_size, seq_len = y.shape
        shifted_y = torch.zeros_like(y)
        y_hat_list = []

        # 自己回帰による逐次推論プロセス
        for t in range(0, seq_len, 2):
            y_hat_step = self(x, shifted_y)
            y_hat_list.append(y_hat_step[:, t:t+2])

            if t + 2 < seq_len:
                shifted_y = shifted_y.clone()
                shifted_y[:, t + 2] = y_hat_step[:, t]
                if t + 3 < seq_len:
                    shifted_y[:, t + 3] = y_hat_step[:, t + 1]

        y_hat = torch.cat(y_hat_list, dim=1)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x, y = x.view(64, -1), y.view(64, -1)

        batch_size, seq_len = y.shape
        shifted_y = torch.zeros_like(y)
        y_hat_list = []

        for t in range(0, seq_len, 2):
            y_hat_step = self(x, shifted_y)
            y_hat_list.append(y_hat_step[:, t:t+2])

            if t + 2 < seq_len:
                shifted_y = shifted_y.clone()
                shifted_y[:, t + 2] = y_hat_step[:, t]
                if t + 3 < seq_len:
                    shifted_y[:, t + 3] = y_hat_step[:, t + 1]

        y_hat = torch.cat(y_hat_list, dim=1)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)
        return loss