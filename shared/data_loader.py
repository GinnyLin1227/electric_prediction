"""
共用資料載入模組 - 模型組三人共用
提供:
  - get_dataloaders(): 取得 train/val/test DataLoader
  - get_scaler(): 取得反標準化用的統計量
  - inverse_y(): 反標準化 y 回原始 kW 尺度
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from shared import config


class PowerDataset(Dataset):
    """
    家庭用電資料集

    X shape: (N, 60, 7)  - 已標準化
    y shape: (N, 60)     - 已標準化
    """
    def __init__(self, X_path, y_path):
        # 用 mmap_mode 避免一次把整份資料塞進記憶體
        self.X = np.load(X_path, mmap_mode='r')
        self.y = np.load(y_path, mmap_mode='r')
        assert len(self.X) == len(self.y), \
            f"X and y length mismatch: {len(self.X)} vs {len(self.y)}"

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 從 mmap 取出後轉 tensor (要 copy 一份避免 mmap 問題)
        x = torch.from_numpy(self.X[idx].copy()).float()
        y = torch.from_numpy(self.y[idx].copy()).float()
        return x, y


def get_dataloaders(data_dir=None, batch_size=None, num_workers=None):
    """
    取得 train / val / test 三個 DataLoader

    回傳:
        train_loader, val_loader, test_loader
    """
    data_dir = data_dir or config.DATA_DIR
    batch_size = batch_size or config.BATCH_SIZE
    num_workers = num_workers if num_workers is not None else config.NUM_WORKERS

    train_set = PowerDataset(
        os.path.join(data_dir, "X_train.npy"),
        os.path.join(data_dir, "y_train.npy"),
    )
    val_set = PowerDataset(
        os.path.join(data_dir, "X_val.npy"),
        os.path.join(data_dir, "y_val.npy"),
    )
    test_set = PowerDataset(
        os.path.join(data_dir, "X_test.npy"),
        os.path.join(data_dir, "y_test.npy"),
    )

    print(f"Train: {len(train_set):,} samples")
    print(f"Val:   {len(val_set):,} samples")
    print(f"Test:  {len(test_set):,} samples")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )

    return train_loader, val_loader, test_loader


def get_scaler(data_dir=None):
    """
    載入 scaler_stats.npz

    回傳 dict, keys: x_mean, x_std, y_mean, y_std
    """
    data_dir = data_dir or config.DATA_DIR
    scaler = np.load(os.path.join(data_dir, "scaler_stats.npz"))
    return {
        'x_mean': scaler['x_mean'],   # shape (7,)
        'x_std':  scaler['x_std'],    # shape (7,)
        'y_mean': scaler['y_mean'],   # shape (60,)
        'y_std':  scaler['y_std'],    # shape (60,)
    }


def inverse_y(y_scaled, scaler):
    """
    把標準化的 y 還原成原始 kW 尺度

    參數:
        y_scaled: numpy array or tensor, shape (N, 60)
        scaler: get_scaler() 的回傳值

    回傳: numpy array, shape (N, 60), 單位 kW
    """
    if isinstance(y_scaled, torch.Tensor):
        y_scaled = y_scaled.detach().cpu().numpy()
    return y_scaled * scaler['y_std'] + scaler['y_mean']


def set_seed(seed=None):
    """
    固定所有隨機性,確保三個模型可以重現
    三人訓練腳本一開始都要呼叫這個
    """
    seed = seed or config.SEED
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 提高重現性 (會稍微犧牲速度)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
