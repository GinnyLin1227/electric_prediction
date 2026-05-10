# Stacked LSTM/GRU 說明

## 我負責的內容

這一版是 Stacked LSTM/GRU 模型，目標是探索「網路深度」與「LSTM vs GRU cell」對多步時間序列預測的影響。

我主要負責：

- `models/stacked_lstm_gru.py`：Stacked LSTM/GRU 模型本體
- `train/train_stacked.py`：模型訓練與評估流程
- `results/stacked_lstm/`、`results/stacked_gru/`：訓練後輸出結果

## 模型設計

Stacked LSTM/GRU 的設計原則是：

> 在 baseline 之上加深網路,並比較 LSTM 與 GRU 兩種 cell 的差異

模型架構如下:

#### 1. Stacked RNN (LSTM 或 GRU)

使用 2 層堆疊的 RNN, 層間加上 dropout 防止過擬合。

- LSTM 版本：採用三閘控機制(input/forget/output gate)
- GRU 版本：採用兩閘控機制(reset/update gate),參數較少

兩種 cell 都能學習:長期用電趨勢、時間依賴關係、序列動態變化。

#### 2. 取最後時間步的 hidden state

從第二層 RNN 的輸出中, 取最後一個時間步的 hidden state 作為「壓縮後的歷史資訊」。

#### 3. Two-layer MLP Output

最後接兩層 FC + ReLU + Dropout, 直接輸出未來 60 分鐘預測:

```
RNN output (last step) → FC(64 → 64) → ReLU → Dropout(0.2) → FC(64 → 60)
```

#### 模型輸入輸出

- 輸入:(batch, 60, 7):過去 60 分鐘共 7 個特徵
- 輸出:(batch, 60):未來 60 分鐘用電預測
- 模型本體:`models/stacked_lstm_gru.py`

## 訓練流程

Stacked LSTM/GRU 與其他兩個模型共用同一套訓練流程, 確保比較公平。

- 資料來源:`data/`
- 共用設定:`shared/config.py`
- DataLoader:`shared/data_loader.py`
- 評估與畫圖:`shared/evaluate.py`
- 訓練入口:`train/train_stacked.py`

## 目前訓練設定

- `EPOCHS = 50`
- `BATCH_SIZE = 256`
- `LEARNING_RATE = 1e-3`
- `WEIGHT_DECAY = 1e-5`
- `GRAD_CLIP_NORM = 1.0`
- `EARLY_STOPPING_PATIENCE = 5`

並使用:Early Stopping、ReduceLROnPlateau Scheduler、Gradient Clipping 提升訓練穩定性。

### 模型架構摘要

目前模型結構(以 stacked_lstm 為例):

```
LSTM(input=7, hidden=64, num_layers=2, dropout=0.2)
→ 取最後時間步 hidden state
→ FC(64 → 64)
→ ReLU
→ Dropout(0.2)
→ FC(64 → 60)
```

總參數量:

- stacked_lstm:60,028 parameters
- stacked_gru:47,036 parameters(比 LSTM 少 22%)

## 實際跑出的結果

目前已完整訓練與驗證成功, 兩個 cell 變體皆已測試。

#### Stacked LSTM

- best epoch:3
- early stopping:epoch 8 triggered
- test metrics:

| Metric | Result |
| --- | --- |
| MAE | 0.3944 kW |
| RMSE | 0.6228 kW |
| MAPE | 62.58% |
| R² | 0.4777 |

- 不同 horizon 的 MAE:

| Horizon | MAE |
| --- | --- |
| step1 | 0.1339 |
| step15 | 0.3244 |
| step30 | 0.4170 |
| step45 | 0.4780 |
| step60 | 0.5245 |

- 詳細數值可看:`results/stacked_lstm/metrics.csv`

#### Stacked GRU

- best epoch:2
- early stopping:epoch 7 triggered
- test metrics:

| Metric | Result |
| --- | --- |
| MAE | 0.4053 kW |
| RMSE | 0.6218 kW |
| MAPE | 66.51% |
| R² | 0.4793 |

- 不同 horizon 的 MAE:

| Horizon | MAE |
| --- | --- |
| step1 | 0.1396 |
| step15 | 0.3336 |
| step30 | 0.4377 |
| step45 | 0.4875 |
| step60 | 0.5284 |

- 詳細數值可看:`results/stacked_gru/metrics.csv`

## 結果分析

從結果可以觀察到:

#### 1. 加深網路相較 baseline 確實提升表現

從 baseline 的 R² 0.4569 提升到 stacked 的 0.4777~0.4793, R² 上升約 4.8%。
證明多一層 RNN 對學習時序模式有實質幫助。

#### 2. GRU 以更少參數達到等效甚至更好的表現

stacked_gru 比 stacked_lstm 少 22% 參數(47K vs 60K), 但 R² 略勝(0.4793 vs 0.4777)、RMSE 也略優(0.6218 vs 0.6228)。
驗證了 GRU 設計者的論點:簡化的閘控機制在許多任務上不輸 LSTM。

#### 3. 預測越遠誤差越大

隨 forecast horizon 增加:MAE 從 step1 的 0.13 一路上升到 step60 的 0.52, 增長約 4 倍。
這是多步時間序列預測中常見現象, 反映遠期預測的固有困難度。

#### 4. 模型在前 2~3 epoch 即達到最佳

兩個變體的 best epoch 都很早(LSTM:3, GRU:2), 之後 val loss 開始上升。
這暗示資料中「容易學的部分」很快被擬合, 剩下的部分屬於外部不可觀測因素造成的固有噪音。

### 輸出內容

訓練完成後, 結果會自動存到:`results/stacked_lstm/` 與 `results/stacked_gru/`

包含:`best.pt`、`metrics.csv`、`history.json`、`loss_curve.png`、`predictions.png`、`horizon_errors.png`

## 執行方式

訓練 Stacked LSTM:

```bash
python train/train_stacked.py --variant stacked_lstm
```

訓練 Stacked GRU:

```bash
python train/train_stacked.py --variant stacked_gru
```

## 比較原則

為了讓三個模型能公平比較, 統一以下設定:

- 同一份資料切分
- 同一套前處理與反標準化
- 同一組訓練超參數
- 同一套評估指標
- 同一個 early stopping 規則

## 補充

- Stacked architecture:加深網路, 提升時序模式學習能力
- LSTM vs GRU:在等效表現下, GRU 提供更輕量的選擇
- Dropout (0.2):層間與 FC 後皆加入, 防止過擬合
