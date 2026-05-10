# Baseline LSTM 說明

## 我負責的內容

這一版是組別 B 的 Baseline LSTM，目標是提供三個模型之間最基本、最公平的比較基準。

我主要負責：

- `models/baseline_lstm.py`：Baseline LSTM 模型本體
- `train/train_baseline.py`：Baseline 的訓練與評估流程
- `results/baseline_lstm/`：訓練後輸出結果

## 模型設計

Baseline 的設計原則是「簡單、可比較、可重現」。

- 使用單層 LSTM
- 輸入 shape 為 `(batch, 60, 7)`
- 輸出 shape 為 `(batch, 60)`
- 取最後一個時間步的 hidden state
- 再接一個小型 MLP 直接預測未來 60 分鐘

總參數量：26,748 parameters

對應檔案：`models/baseline_lstm.py`

## 訓練流程

Baseline 的訓練流程跟其他兩個模型共用同一套設定，避免比較不公平。

- 資料來源：`data/`
- 共用設定：`shared/config.py`
- DataLoader：`shared/data_loader.py`
- 評估與畫圖：`shared/evaluate.py`

訓練入口：`train/train_baseline.py`

目前設定：

- `EPOCHS = 50`，但會搭配 early stopping
- `BATCH_SIZE = 256`
- `LEARNING_RATE = 1e-3`
- `WEIGHT_DECAY = 1e-5`
- `GRAD_CLIP_NORM = 1.0`
- `EARLY_STOPPING_PATIENCE = 5`

## 實際跑出的結果

目前在 `v8_Xub` 環境中已經完整訓練並驗證成功。

- best epoch：2
- training time：約 3.48 分鐘
- test MAE：0.4071 kW
- test RMSE：0.6350 kW
- test R2：0.4569

不同 horizon 的 MAE：

Horizon	MAE
step1	0.1135
step15	0.3295
step30	0.4388
step45	0.4958
step60	0.5325

詳細數值可看：`results/baseline_lstm/metrics.csv`

## 輸出內容

訓練完成後，結果會自動存到：`results/baseline_lstm/`

包含：

- `best.pt`：最佳模型權重
- `metrics.csv`：測試指標
- `history.json`：訓練歷程
- `loss_curve.png`：訓練/驗證 loss 曲線
- `predictions.png`：預測 vs 實際圖
- `horizon_errors.png`：不同 horizon 的誤差圖

## 執行方式

先切到環境：

```bash
conda activate v8_Xub
```

再開始訓練：

```bash
python train/train_baseline.py
```

## 比較原則

為了讓三個模型能公平比較，大家要盡量統一：

- 同一份資料切分
- 同一套前處理與反標準化
- 同一組訓練超參數
- 同一套評估指標
- 同一個 early stopping 規則

## 補充

我先前也對照過 `stacked_lstm_gru` 的訓練方式，baseline 這版已經沿用相同的訓練流程，只是模型本體改成更簡單的單層 LSTM，方便當作基準線。