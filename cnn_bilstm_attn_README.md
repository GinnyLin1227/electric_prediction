# **CNN-BiLSTM-Attn 說明**



這一版是 CNN-BiLSTM-Attention 模型，目標是提升多步時間序列預測能力。

我主要負責：

* models/cnn\_bilstm\_attn.py：CNN-BiLSTM-Attention 模型本體
* train/train\_cnn\_bilstm.py：模型訓練與評估流程
* results/cnn\_bilstm\_attn/：訓練後輸出結果


模型設計
---

CNN-BiLSTM-Attention 的設計原則是：

同時學習局部特徵、長期時序依賴，以及重要時間步資訊

模型架構如下：

1. ###### CNN Feature Extraction

先使用 1D CNN：Conv1D、BatchNorm、MaxPooling提取短期時間序列特徵。

CNN 能有效學習：短期用電波動、局部時間模式、Temporal local patterns


2. BiLSTM Sequence Learning
---

* CNN 輸出後接雙向 LSTM（BiLSTM）：

forward sequence
backward sequence

同時學習前後時間資訊。

* BiLSTM 能更完整捕捉：

長期用電趨勢
時間依賴關係
Sequence dynamics


3. Attention Mechanism
---

在 BiLSTM 後加入 Attention Layer。

Attention 的作用是：

讓模型自動聚焦於較重要的時間步

避免所有時間點被平均看待。



4. ###### Fully Connected Output

最後經過：FC layer、ReLU、Dropout

直接輸出未來 60 分鐘預測結果。

模型輸入輸出

* 輸入：(batch, 60, 7)：過去 60 分鐘共 7 個特徵
* 輸出：(batch, 60)：未來 60 分鐘用電預測
* 模型本體：models/cnn\_bilstm\_attn.py


訓練流程
---

CNN-BiLSTM-Attention 與其他兩個模型共用同一套訓練流程，確保比較公平。

* 資料來源：data/
* 共用設定：shared/config.py、DataLoader、shared/data\_loader.py
* 評估與畫圖：shared/evaluate.py
* 訓練入口：train/train\_cnn\_bilstm.py


目前訓練設定
---

* EPOCHS = 50
* BATCH\_SIZE = 256
* LEARNING\_RATE = 1e-3
* WEIGHT\_DECAY = 1e-5
* GRAD\_CLIP\_NORM = 1.0
* EARLY\_STOPPING\_PATIENCE = 5

並使用：Early Stopping、ReduceLROnPlateau Scheduler、Gradient Clipping提升訓練穩定性。

### 

### 模型架構摘要

目前模型結構：

Conv1D(7 → 64)
→ BatchNorm1D
→ MaxPool1D
→ BiLSTM(hidden=64, bidirectional=True)
→ Attention Layer
→ FC(128 → 128)
→ Dropout(0.2)
→ FC(128 → 60)

總參數量：92,477 parameters


實際跑出的結果
---

目前已在 dl\_project 環境中完整訓練與驗證成功。

* best epoch：2
* early stopping：epoch 7 triggered
* test metrics：

Metric	Result
MAE	0.4008 kW
RMSE	0.6266 kW
MAPE	64.16%
R²	0.4712

* 不同 horizon 的 MAE：

Horizon	MAE
step1	0.1471
step15	0.3316
step30	0.4179
step45	0.4883
step60	0.5240

* 詳細數值可看：results/cnn\_bilstm\_attn/metrics.csv


結果分析
---

從結果可以觀察到：

1. ###### 模型能有效學習短期與長期用電模式

CNN 能提取局部特徵，BiLSTM 能捕捉長期時序依賴。

2. ###### 預測越遠誤差越大

隨 forecast horizon 增加：MAE 逐漸上升，這是多步時間序列預測中常見現象。

3. ###### Attention 提升重要時間步學習能力

Attention 能讓模型聚焦較關鍵的歷史時間點，避免資訊平均化。



### 輸出內容

訓練完成後，結果會自動存到：results/cnn\_bilstm\_attn/

包含：best.pt、metrics.csv、history.json、loss\_curve.png、predictions.png、horizon\_errors.png


執行方式
---

先切到環境：conda activate dl\_project

再開始訓練：python train/train\_cnn\_bilstm.py


比較原則
---

為了讓三個模型能公平比較，統一以下設定：

* 同一份資料切分
* 同一套前處理與反標準化
* 同一組訓練超參數
* 同一套評估指標
* 同一個 early stopping 規則


補充
---

CNN：提升局部特徵擷取能力
BiLSTM：加強雙向時序學習
Attention：強化重要時間步關注能力

