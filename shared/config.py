"""
共用設定檔 - 模型組三人共用
任何訓練超參數的調整都改這裡，三個模型同步更新
"""
import os

# ============================================================
# 路徑設定（依各自環境調整）
# ============================================================
DATA_DIR = "./data"  # 改成你的資料夾路徑
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# 資料規格（由資料組決定，不要改）
# ============================================================
INPUT_SIZE = 7        # 7 個原始特徵
LOOKBACK = 60         # 過去 60 分鐘
HORIZON = 60          # 預測未來 60 分鐘

# ============================================================
# 訓練超參數（三人統一）
# ============================================================
SEED = 42
BATCH_SIZE = 256
NUM_WORKERS = 0       # DataLoader 並行讀取，依電腦規格調整（Colab 通常設 2）

LEARNING_RATE = 1e-3
EPOCHS = 50           # 上限，搭配 early stopping
WEIGHT_DECAY = 1e-5   # L2 正則化

# Early stopping
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 1e-4

# Learning rate scheduler (ReduceLROnPlateau)
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_MIN_LR = 1e-6

# Gradient clipping (避免梯度爆炸,RNN 常用)
GRAD_CLIP_NORM = 1.0

# ============================================================
# 評估設定
# ============================================================
# 在報告中要分析的特定 horizon 步數(用於誤差隨時間變化分析)
HORIZONS_TO_REPORT = [1, 15, 30, 45, 60]
