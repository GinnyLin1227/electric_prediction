"""
共用評估模組 - 模型組三人共用
提供:
  - compute_metrics(): 計算 MAE / RMSE / MAPE / R² (原始 kW 尺度)
  - evaluate_model(): 跑完整測試集,回傳指標 + 預測值
  - plot_loss_curve(): 訓練/驗證 loss 曲線
  - plot_predictions(): 預測 vs 實際時間序列對比
  - plot_horizon_errors(): 不同 horizon 步數的誤差變化
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from shared import config
from shared.data_loader import inverse_y


def compute_metrics(y_pred, y_true, horizons_to_report=None):
    """
    計算評估指標 (在原始 kW 尺度上)

    參數:
        y_pred: numpy array, shape (N, 60), 單位 kW (已反標準化)
        y_true: numpy array, shape (N, 60), 單位 kW (已反標準化)
        horizons_to_report: list, 例如 [1, 15, 30, 45, 60]

    回傳: dict
    """
    # 整體指標 (所有 horizon 平均)
    err = y_pred - y_true
    mae = np.abs(err).mean()
    rmse = np.sqrt((err ** 2).mean())

    # MAPE - 避免除以 0,只算 y_true 不為 0 的部分
    nonzero = np.abs(y_true) > 1e-3
    mape = np.abs(err[nonzero] / y_true[nonzero]).mean() * 100

    # R²
    ss_res = (err ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot

    metrics = {
        'MAE_kW': float(mae),
        'RMSE_kW': float(rmse),
        'MAPE_%': float(mape),
        'R2': float(r2),
    }

    # 各 horizon 步的 MAE (用於分析)
    if horizons_to_report is None:
        horizons_to_report = config.HORIZONS_TO_REPORT
    for h in horizons_to_report:
        idx = h - 1  # 1-based -> 0-based
        if 0 <= idx < y_pred.shape[1]:
            mae_h = np.abs(y_pred[:, idx] - y_true[:, idx]).mean()
            metrics[f'MAE_kW@step{h}'] = float(mae_h)

    return metrics


@torch.no_grad()
def evaluate_model(model, loader, scaler, device='cuda'):
    """
    在指定 DataLoader 上評估模型

    參數:
        model: 已訓練好的 PyTorch 模型
        loader: 通常是 test_loader
        scaler: get_scaler() 的回傳值
        device: 'cuda' or 'cpu'

    回傳:
        metrics: dict
        y_pred_kW: numpy array (N, 60), 原始 kW 尺度
        y_true_kW: numpy array (N, 60), 原始 kW 尺度
    """
    model.eval()
    preds, trues = [], []
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y_pred = model(X).cpu().numpy()
        preds.append(y_pred)
        trues.append(y.numpy())

    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(trues, axis=0)

    # 反標準化回 kW
    y_pred_kW = inverse_y(y_pred, scaler)
    y_true_kW = inverse_y(y_true, scaler)

    metrics = compute_metrics(y_pred_kW, y_true_kW)
    return metrics, y_pred_kW, y_true_kW


def plot_loss_curve(history, save_path=None, title="Training Curve"):
    """
    畫訓練/驗證 loss 曲線

    參數:
        history: dict, keys: 'train_loss', 'val_loss', 各為 list
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE on standardized y)')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"Saved: {save_path}")
    plt.close()


def plot_predictions(y_pred_kW, y_true_kW, sample_indices=None,
                     save_path=None, title="Prediction vs Actual"):
    """
    畫幾個樣本的預測 vs 實際時間序列對比

    參數:
        y_pred_kW, y_true_kW: shape (N, 60), 原始 kW
        sample_indices: 要展示哪幾筆,預設 [0, N//4, N//2, 3*N//4]
    """
    if sample_indices is None:
        n = len(y_pred_kW)
        sample_indices = [0, n // 4, n // 2, 3 * n // 4]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, idx in zip(axes.flat, sample_indices):
        ax.plot(y_true_kW[idx], label='Actual', linewidth=2)
        ax.plot(y_pred_kW[idx], label='Predicted', linewidth=2, linestyle='--')
        ax.set_xlabel('Future Minute')
        ax.set_ylabel('Power (kW)')
        ax.set_title(f'Sample #{idx}')
        ax.legend()
        ax.grid(alpha=0.3)
    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"Saved: {save_path}")
    plt.close()


def plot_horizon_errors(y_pred_kW, y_true_kW, save_path=None,
                        title="MAE by Forecast Horizon"):
    """
    畫每個 horizon 步的 MAE (分析誤差是否隨時間變大)
    """
    mae_per_step = np.abs(y_pred_kW - y_true_kW).mean(axis=0)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(mae_per_step) + 1), mae_per_step, linewidth=2)
    plt.xlabel('Forecast Horizon (minutes ahead)')
    plt.ylabel('MAE (kW)')
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"Saved: {save_path}")
    plt.close()


def print_metrics(metrics, model_name=""):
    """漂亮地印出指標"""
    print(f"\n{'=' * 50}")
    print(f"  Evaluation Results{' - ' + model_name if model_name else ''}")
    print(f"{'=' * 50}")
    for k, v in metrics.items():
        if 'MAPE' in k:
            print(f"  {k:<20s}: {v:.2f}%")
        else:
            print(f"  {k:<20s}: {v:.4f}")
    print(f"{'=' * 50}\n")


def save_metrics(metrics, save_path):
    """把指標存成 csv 方便整理"""
    import csv
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for k, v in metrics.items():
            writer.writerow([k, v])
    print(f"Saved: {save_path}")
