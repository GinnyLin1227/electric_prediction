"""
Stacked LSTM / GRU 模型
- 透過 cell_type 參數切換 LSTM 或 GRU
- 多層堆疊,層間 dropout
- 取最後一個時間步的 hidden state 接全連接層輸出 60 步
"""
import torch
import torch.nn as nn


class StackedRNN(nn.Module):
    """
    Stacked LSTM / GRU for multi-step time series forecasting

    參數:
        input_size: 輸入特徵數 (=7)
        hidden_size: 每層 hidden units 數
        num_layers: 堆疊層數
        horizon: 預測未來幾步 (=60)
        dropout: 層間 dropout 比率 (只在 num_layers >= 2 時生效)
        cell_type: 'LSTM' or 'GRU'
        bidirectional: 是否雙向
    """
    def __init__(
        self,
        input_size=7,
        hidden_size=64,
        num_layers=2,
        horizon=60,
        dropout=0.2,
        cell_type='LSTM',
        bidirectional=False,
    ):
        super().__init__()
        self.cell_type = cell_type.upper()
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        rnn_class = {'LSTM': nn.LSTM, 'GRU': nn.GRU}[self.cell_type]
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        out_dim = hidden_size * (2 if bidirectional else 1)

        # 兩層 FC,中間 dropout 防止過擬合
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
        # out: (batch, lookback, hidden * num_directions)
        out, _ = self.rnn(x)
        # 取最後一個時間步
        last = out[:, -1, :]
        # 直接輸出 60 步預測
        return self.fc(last)


def build_model(variant='stacked_lstm'):
    """
    依 variant 名稱建立模型 - 方便做消融實驗

    支援:
        'stacked_lstm':  2 層 LSTM,  hidden=64
        'stacked_gru':   2 層 GRU,   hidden=64
        'deep_lstm':     3 層 LSTM,  hidden=128
        'deep_gru':      3 層 GRU,   hidden=128
    """
    configs = {
        'stacked_lstm': dict(cell_type='LSTM', num_layers=2, hidden_size=64),
        'stacked_gru':  dict(cell_type='GRU',  num_layers=2, hidden_size=64),
        'deep_lstm':    dict(cell_type='LSTM', num_layers=3, hidden_size=128),
        'deep_gru':     dict(cell_type='GRU',  num_layers=3, hidden_size=128),
    }
    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(configs.keys())}")
    return StackedRNN(**configs[variant])


if __name__ == "__main__":
    # 簡單測試: 確認 forward 跑得通、shape 對
    for variant in ['stacked_lstm', 'stacked_gru', 'deep_lstm', 'deep_gru']:
        model = build_model(variant)
        n_params = sum(p.numel() for p in model.parameters())
        x = torch.randn(8, 60, 7)
        y = model(x)
        print(f"{variant:<15s} params={n_params:>8,}  output={tuple(y.shape)}")
