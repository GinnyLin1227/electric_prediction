"""
Baseline LSTM 模型
- 單層 LSTM 作為最基準的序列預測模型
- 取最後一個時間步的 hidden state 接全連接層輸出 60 步
"""
import torch
import torch.nn as nn


class BaselineLSTM(nn.Module):
    """
    Baseline LSTM for multi-step time series forecasting

    參數:
        input_size: 輸入特徵數 (=7)
        hidden_size: hidden units 數
        num_layers: LSTM 層數，baseline 預設 1
        horizon: 預測未來幾步 (=60)
        dropout: FC 中的 dropout 比率
        bidirectional: 是否雙向
    """
    def __init__(
        self,
        input_size=7,
        hidden_size=64,
        num_layers=1,
        horizon=60,
        dropout=0.2,
        bidirectional=False,
    ):
        super().__init__()
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        out_dim = hidden_size * (2 if bidirectional else 1)

        # 單層 LSTM 後接簡單 MLP，維持 baseline 的可比性
        self.fc = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, horizon),
        )

    def forward(self, x):
        """
        x: (batch, 60, 7)
        return: (batch, 60)
        """
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


def build_model(variant='baseline_lstm'):
    """
    依 variant 名稱建立模型

    目前支援:
        'baseline_lstm': 單層 LSTM, hidden=64
    """
    configs = {
        'baseline_lstm': dict(hidden_size=64, num_layers=1, bidirectional=False),
    }
    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(configs.keys())}")
    return BaselineLSTM(**configs[variant])


if __name__ == "__main__":
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(8, 60, 7)
    y = model(x)
    print(f"baseline_lstm    params={n_params:>8,}  output={tuple(y.shape)}")