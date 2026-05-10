"""
Stacked LSTM/GRU 訓練腳本

用法:
    python train_stacked.py --variant stacked_lstm
    python train_stacked.py --variant stacked_gru
    python train_stacked.py --variant deep_lstm
"""
import os
import sys
import time
import argparse
import json

import torch
import torch.nn as nn

# 加入路徑讓 import 找得到 shared/ 和 models/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'shared')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

import config
from data_loader import get_dataloaders, get_scaler, set_seed
from evaluate import (
    evaluate_model, plot_loss_curve, plot_predictions,
    plot_horizon_errors, print_metrics, save_metrics,
)
from stacked_lstm_gru import build_model


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n = 0.0, 0
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
        optimizer.step()

        bs = X.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y_pred = model(X)
        loss = criterion(y_pred, y)
        bs = X.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / n


def main(args):
    # ---------------- 設定 ----------------
    set_seed()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Variant: {args.variant}")

    # 結果存放資料夾
    run_dir = os.path.join(config.RESULTS_DIR, args.variant)
    os.makedirs(run_dir, exist_ok=True)

    # ---------------- 資料 ----------------
    train_loader, val_loader, test_loader = get_dataloaders()
    scaler = get_scaler()

    # ---------------- 模型 ----------------
    model = build_model(args.variant).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE,
        min_lr=config.LR_SCHEDULER_MIN_LR,
    )
    criterion = nn.MSELoss()

    # ---------------- 訓練迴圈 ----------------
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val = float('inf')
    best_epoch = -1
    bad_epochs = 0
    best_ckpt = os.path.join(run_dir, 'best.pt')

    print(f"\n{'=' * 60}")
    print(f"  Training: {args.variant}")
    print(f"{'=' * 60}")
    t0 = time.time()

    for epoch in range(1, config.EPOCHS + 1):
        ep_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        cur_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(cur_lr)

        ep_time = time.time() - ep_start
        improved = val_loss < best_val - config.EARLY_STOPPING_MIN_DELTA
        marker = " *" if improved else ""
        print(f"Epoch {epoch:3d}/{config.EPOCHS} | "
              f"train={train_loss:.6f} | val={val_loss:.6f} | "
              f"lr={cur_lr:.2e} | {ep_time:.1f}s{marker}")

        if improved:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            bad_epochs += 1
            if bad_epochs >= config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best epoch={best_epoch}, val={best_val:.6f})")
                break

    total_time = time.time() - t0
    print(f"\nTraining finished in {total_time/60:.1f} min")
    print(f"Best epoch: {best_epoch}, val_loss: {best_val:.6f}")

    # ---------------- 載入最佳 checkpoint 測試 ----------------
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    metrics, y_pred_kW, y_true_kW = evaluate_model(model, test_loader, scaler, device)

    # 加入訓練資訊
    metrics['n_params'] = n_params
    metrics['best_epoch'] = best_epoch
    metrics['training_time_min'] = round(total_time / 60, 2)

    print_metrics(metrics, model_name=args.variant)
    save_metrics(metrics, os.path.join(run_dir, 'metrics.csv'))

    # 訓練紀錄存 json (方便後面分析)
    with open(os.path.join(run_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # ---------------- 視覺化 ----------------
    plot_loss_curve(
        history,
        save_path=os.path.join(run_dir, 'loss_curve.png'),
        title=f"{args.variant} - Training Curve",
    )
    plot_predictions(
        y_pred_kW, y_true_kW,
        save_path=os.path.join(run_dir, 'predictions.png'),
        title=f"{args.variant} - Predictions",
    )
    plot_horizon_errors(
        y_pred_kW, y_true_kW,
        save_path=os.path.join(run_dir, 'horizon_errors.png'),
        title=f"{args.variant} - MAE by Horizon",
    )

    print(f"\nAll outputs saved to: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--variant', type=str, default='stacked_lstm',
        choices=['stacked_lstm', 'stacked_gru', 'deep_lstm', 'deep_gru'],
        help='Model variant to train',
    )
    args = parser.parse_args()
    main(args)
