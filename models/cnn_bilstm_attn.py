import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()

        self.attention = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_output):
        """
        lstm_output:
        (batch_size, seq_len, hidden_dim*2)
        """

        # attention score
        attn_weights = self.attention(lstm_output)

        # softmax over time dimension
        attn_weights = torch.softmax(attn_weights, dim=1)

        # weighted sum
        context = torch.sum(attn_weights * lstm_output, dim=1)

        return context, attn_weights


class CNN_BiLSTM_Attention(nn.Module):
    """
    CNN + BiLSTM + Attention
    for multi-step household power forecasting
    """

    def __init__(
        self,
        input_dim=7,
        cnn_channels=64,
        lstm_hidden=64,     # 與 baseline 對齊
        lstm_layers=1,      # 與 baseline 對齊
        output_dim=60,
        dropout=0.2
    ):
        super(CNN_BiLSTM_Attention, self).__init__()

        # =========================
        # CNN Feature Extractor
        # =========================

        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=cnn_channels,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm1d(cnn_channels)

        self.pool = nn.MaxPool1d(kernel_size=2)

        # =========================
        # BiLSTM
        # =========================

        self.bilstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0  # 單層 LSTM 不使用 dropout
        )

        # =========================
        # Attention
        # =========================

        self.attention = AttentionLayer(lstm_hidden)

        # =========================
        # Fully Connected
        # =========================

        self.fc1 = nn.Linear(lstm_hidden * 2, lstm_hidden * 2)

        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(lstm_hidden * 2, output_dim)

    def forward(self, x):
        """
        x:
        (batch_size, 60, 7)

        return:
        (batch_size, 60)
        """

        # CNN expects:
        # (batch, channels, seq_len)

        x = x.permute(0, 2, 1)

        x = self.conv1(x)

        x = self.bn1(x)

        x = F.relu(x)

        x = self.pool(x)

        # Back to:
        # (batch, seq_len, features)

        x = x.permute(0, 2, 1)

        # BiLSTM
        lstm_out, _ = self.bilstm(x)

        # Attention
        context, attn_weights = self.attention(lstm_out)

        # FC
        x = self.fc1(context)

        x = F.relu(x)

        x = self.dropout(x)

        output = self.fc2(x)

        return output


if __name__ == "__main__":

    model = CNN_BiLSTM_Attention()

    n_params = sum(p.numel() for p in model.parameters())

    x = torch.randn(8, 60, 7)

    y = model(x)

    print(f"cnn_bilstm_attn params={n_params:,}")
    print("output shape:", y.shape)