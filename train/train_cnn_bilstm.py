import os
import sys
import json
import torch
import torch.nn as nn

# 讓 train 可以找到 models/ 與 shared/
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models.cnn_bilstm_attn import CNN_BiLSTM_Attention

from shared import config

from shared.data_loader import (
    get_dataloaders,
    get_scaler,
    set_seed
)

from shared.evaluate import (
    evaluate_model,
    plot_loss_curve,
    plot_predictions,
    plot_horizon_errors,
    print_metrics,
    save_metrics
)


# ==========================================================
# Set Seed
# ==========================================================

set_seed(config.SEED)


# ==========================================================
# Device
# ==========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ==========================================================
# Result Directory
# ==========================================================

RESULT_DIR = os.path.join(
    config.RESULTS_DIR,
    "cnn_bilstm_attn"
)

os.makedirs(RESULT_DIR, exist_ok=True)


# ==========================================================
# DataLoader
# ==========================================================

train_loader, val_loader, test_loader = get_dataloaders()

scaler = get_scaler()


# ==========================================================
# Model
# ==========================================================

model = CNN_BiLSTM_Attention().to(device)

print(model)

n_params = sum(p.numel() for p in model.parameters())

print(f"Total Parameters: {n_params:,}")


# ==========================================================
# Loss Function
# ==========================================================

criterion = nn.MSELoss()


# ==========================================================
# Optimizer
# ==========================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY
)


# ==========================================================
# Learning Rate Scheduler
# ==========================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=config.LR_SCHEDULER_FACTOR,
    patience=config.LR_SCHEDULER_PATIENCE,
    min_lr=config.LR_SCHEDULER_MIN_LR
)


# ==========================================================
# Training Settings
# ==========================================================

best_val_loss = float("inf")

patience_counter = 0

history = {
    "train_loss": [],
    "val_loss": []
}


# ==========================================================
# Training Loop
# ==========================================================

for epoch in range(config.EPOCHS):

    # ======================================================
    # Train
    # ======================================================

    model.train()

    train_loss = 0.0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.GRAD_CLIP_NORM
        )

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ======================================================
    # Validation
    # ======================================================

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    # ======================================================
    # Scheduler Step
    # ======================================================

    scheduler.step(val_loss)

    # ======================================================
    # Save History
    # ======================================================

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    # ======================================================
    # Print
    # ======================================================

    current_lr = optimizer.param_groups[0]['lr']

    print(
        f"Epoch [{epoch+1}/{config.EPOCHS}] "
        f"Train Loss: {train_loss:.6f} "
        f"Val Loss: {val_loss:.6f} "
        f"LR: {current_lr:.6e}"
    )

    # ======================================================
    # Save Best Model
    # ======================================================

    if val_loss < best_val_loss - config.EARLY_STOPPING_MIN_DELTA:

        best_val_loss = val_loss

        patience_counter = 0

        torch.save(
            model.state_dict(),
            os.path.join(RESULT_DIR, "best.pt")
        )

        print("Best model saved!")

    else:

        patience_counter += 1

        print(
            f"EarlyStopping Counter: "
            f"{patience_counter}/"
            f"{config.EARLY_STOPPING_PATIENCE}"
        )

    # ======================================================
    # Early Stopping
    # ======================================================

    if patience_counter >= config.EARLY_STOPPING_PATIENCE:

        print("Early stopping triggered!")

        break


# ==========================================================
# Training Finished
# ==========================================================

print("Training Finished!")


# ==========================================================
# Load Best Model
# ==========================================================

model.load_state_dict(
    torch.load(
        os.path.join(RESULT_DIR, "best.pt")
    )
)


# ==========================================================
# Evaluate on Test Set
# ==========================================================

metrics, y_pred_kW, y_true_kW = evaluate_model(
    model,
    test_loader,
    scaler,
    device=device
)

print_metrics(metrics, "CNN_BiLSTM_Attention")


# ==========================================================
# Save Metrics
# ==========================================================

save_metrics(
    metrics,
    os.path.join(RESULT_DIR, "metrics.csv")
)


# ==========================================================
# Save History
# ==========================================================

with open(
    os.path.join(RESULT_DIR, "history.json"),
    "w"
) as f:

    json.dump(history, f, indent=4)

print("Saved: history.json")


# ==========================================================
# Plot Loss Curve
# ==========================================================

plot_loss_curve(
    history,
    save_path=os.path.join(
        RESULT_DIR,
        "loss_curve.png"
    ),
    title="CNN-BiLSTM-Attention Training Curve"
)


# ==========================================================
# Plot Predictions
# ==========================================================

plot_predictions(
    y_pred_kW,
    y_true_kW,
    save_path=os.path.join(
        RESULT_DIR,
        "predictions.png"
    ),
    title="CNN-BiLSTM-Attention Predictions"
)


# ==========================================================
# Plot Horizon Errors
# ==========================================================

plot_horizon_errors(
    y_pred_kW,
    y_true_kW,
    save_path=os.path.join(
        RESULT_DIR,
        "horizon_errors.png"
    ),
    title="CNN-BiLSTM-Attention Horizon Errors"
)


print("All results saved successfully!")