import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from mlflow.models.signature import infer_signature
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import optuna

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report
)

from dataclasses import dataclass

from steps.data import CharsDataset, transform_ocr, NUM_CLASSES, DATASET_DIR, CSV_PATH


# ===========================================================
# MODEL
# ===========================================================
class BetterCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ===========================================================
# DATASET
# ===========================================================
def preparar_dataset(csv_path, root=DATASET_DIR, val_split=0.2, seed=42):
    df = pd.read_csv(csv_path)
    dataset = CharsDataset(df, root_dir=root, transform=transform_ocr)

    val_len = int(len(dataset) * val_split)
    train_len = len(dataset) - val_len

    g = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_len, val_len], generator=g)


# ===========================================================
# METRICS
# ===========================================================
def topk_accuracy(probs, y_true, k=3):
    # probs: (N, C)
    topk = np.argsort(probs, axis=1)[:, -k:]
    y_true = np.asarray(y_true)
    return float(np.mean([y_true[i] in topk[i] for i in range(len(y_true))]))


def compute_metrics(labels, preds, probs, num_classes):
    labels = np.asarray(labels)
    preds = np.asarray(preds)

    acc = accuracy_score(labels, preds)
    bacc = balanced_accuracy_score(labels, preds)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    top3 = topk_accuracy(probs, labels, k=3)
    top5 = topk_accuracy(probs, labels, k=5) if num_classes >= 5 else None

    auc_ovr = None
    try:
        auc_ovr = roc_auc_score(labels, probs, multi_class="ovr")
    except Exception:
        pass

    metrics = {
        "val_acc": float(acc),
        "val_balanced_acc": float(bacc),
        "val_precision_macro": float(p_macro),
        "val_recall_macro": float(r_macro),
        "val_f1_macro": float(f1_macro),
        "val_precision_weighted": float(p_weight),
        "val_recall_weighted": float(r_weight),
        "val_f1_weighted": float(f1_weight),
        "val_top3_acc": float(top3),
    }
    if top5 is not None:
        metrics["val_top5_acc"] = float(top5)
    if auc_ovr is not None:
        metrics["val_auc_ovr"] = float(auc_ovr)

    return metrics


# ===========================================================
# TRAIN / VALIDATE
# ===========================================================
def train_one_epoch(model, loader, device, opt, crit, scaler=None, use_amp=False, grad_clip=None):
    model.train()
    total_loss, correct = 0.0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        opt.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast(enabled=True):
                out = model(imgs)
                loss = crit(out, labels)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            out = model(imgs)
            loss = crit(out, labels)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        total_loss += float(loss.item())
        correct += int((out.argmax(1) == labels).sum().item())

    return total_loss / len(loader), correct / len(loader.dataset)


def validar(model, loader, device, crit):
    model.eval()
    total_loss = 0.0
    labels_all, preds_all, probs_all = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = crit(out, labels)
            total_loss += float(loss.item())

            probs = torch.softmax(out, dim=1)
            preds = probs.argmax(1)

            probs_all.append(probs.cpu().numpy())
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    probs_all = np.vstack(probs_all)
    val_loss = total_loss / len(loader)
    return val_loss, labels_all, preds_all, probs_all


# ===========================================================
# ARTIFACTS
# ===========================================================
def guardar_confusion(labels, preds, path, normalize=False):
    cm = confusion_matrix(labels, preds)
    if normalize:
        cm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)

    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, cmap="Blues")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def guardar_curvas(history, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loss curve
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.legend()
    plt.title("Loss")
    loss_path = out_dir / "loss_curve.png"
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.legend()
    plt.title("Accuracy")
    acc_path = out_dir / "acc_curve.png"
    plt.tight_layout()
    plt.savefig(acc_path)
    plt.close()

    return loss_path, acc_path


def guardar_classification_report(labels, preds, path: Path):
    report = classification_report(labels, preds, zero_division=0)
    path.write_text(report, encoding="utf-8")
    return path


# ===========================================================
# CONFIG
# ===========================================================
@dataclass
class TrainConfig:
    epochs: int = 40
    batch: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 6
    grad_clip: float = 1.0
    val_split: float = 0.2
    seed: int = 42
    label_smoothing: float = 0.0


# ===========================================================
# TRAIN MAIN (single run)
# ===========================================================
def train_main(config: TrainConfig):
    with mlflow.start_run(run_name="train", nested=True):
        mlflow.log_params(vars(config))

        # Data
        train_ds, val_ds = preparar_dataset(CSV_PATH, val_split=config.val_split, seed=config.seed)
        train_loader = DataLoader(train_ds, batch_size=config.batch, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.batch)

        # Model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = BetterCNN().to(device)

        crit = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        opt = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)

        use_amp = (device == "cuda")
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        best_score = -1.0
        best_state = None
        best_epoch = -1
        bad_epochs = 0

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        last_y_true, last_y_pred = None, None

        for ep in range(config.epochs):
            t0 = time.time()

            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, device, opt, crit,
                scaler=scaler, use_amp=use_amp, grad_clip=config.grad_clip
            )

            val_loss, y_true, y_pred, y_prob = validar(model, val_loader, device, crit)
            metrics = compute_metrics(y_true, y_pred, y_prob, NUM_CLASSES)

            # history
            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(metrics["val_acc"])

            # log
            mlflow.log_metric("train_loss", tr_loss, step=ep)
            mlflow.log_metric("train_acc", tr_acc, step=ep)
            mlflow.log_metric("val_loss", val_loss, step=ep)
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v), step=ep)

            mlflow.log_metric("lr", float(opt.param_groups[0]["lr"]), step=ep)
            mlflow.log_metric("epoch_time_s", float(time.time() - t0), step=ep)

            score = metrics["val_f1_macro"]
            scheduler.step(score)

            print(
                f"[Epoch {ep}] tr_acc={tr_acc:.3f} "
                f"val_acc={metrics['val_acc']:.3f} val_f1m={metrics['val_f1_macro']:.3f} "
                f"top3={metrics['val_top3_acc']:.3f} lr={opt.param_groups[0]['lr']:.2e}"
            )

            improved = score > (best_score + 1e-6)
            
            if improved:
                best_score = score
                best_epoch = ep
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
                print(
                    f"[Epoch {ep}] ✅ Improved val_f1_macro={best_score:.4f} "
                    f"(val_acc={metrics['val_acc']:.4f})"
                )
            else:
                bad_epochs += 1
                print(
                    f"[Epoch {ep}] ❌ No improvement: val_f1_macro={score:.4f} "
                    f"(best={best_score:.4f} @ epoch {best_epoch}) "
                    f"| no_improve={bad_epochs}/{config.patience}"
                )
            
                if bad_epochs >= config.patience:
                    print(
                        f"🛑 Early stopping. Best val_f1_macro={best_score:.4f} "
                        f"at epoch {best_epoch}. Stopped at epoch {ep}."
                    )
                    mlflow.log_param("early_stopping_triggered", True)
                    mlflow.log_param("early_stopped_epoch", int(ep))
                    mlflow.log_param("best_epoch", int(best_epoch))
                    mlflow.log_metric("best_val_f1_macro", float(best_score))
                    mlflow.log_metric("no_improve_epochs", int(bad_epochs))
                    last_y_true, last_y_pred = y_true, y_pred
                    break

            last_y_true, last_y_pred = y_true, y_pred

        # Restore best
        if best_state is not None:
            model.load_state_dict(best_state)

        # Artifacts
        out_dir = Path("artifacts/modelado")
        out_dir.mkdir(parents=True, exist_ok=True)

        # curves
        curves_dir = out_dir / "curves"
        loss_path, acc_path = guardar_curvas(history, curves_dir)
        mlflow.log_artifact(str(loss_path))
        mlflow.log_artifact(str(acc_path))

        # confusion matrices
        cm_path = guardar_confusion(last_y_true, last_y_pred, out_dir / "confusion_matrix.png", normalize=False)
        cmn_path = guardar_confusion(last_y_true, last_y_pred, out_dir / "confusion_matrix_norm.png", normalize=True)
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(cmn_path))

        # classification report
        cr_path = guardar_classification_report(last_y_true, last_y_pred, out_dir / "classification_report.txt")
        mlflow.log_artifact(str(cr_path))

        # Log model
        input_example_tensor = torch.zeros((1, 1, 32, 32), device=device)
        input_example = input_example_tensor.detach().cpu().numpy()
        pred_example = model(input_example_tensor).detach().cpu().numpy()
        signature = infer_signature(input_example, pred_example)

        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="ocr-model",
            signature=signature,
            input_example=input_example,
        )

        # final metric for convenience
        mlflow.log_param("best_epoch", int(best_epoch))
        mlflow.log_metric("best_val_f1_macro", float(best_score))
        
        return model, float(best_score), int(best_epoch)



# ===========================================================
# OPTUNA HPO
# ===========================================================
def build_objective(
    epochs: int,
    patience: int,
    val_split: float = 0.2,
    seed: int = 42,
):
    def objective(trial):
        cfg = TrainConfig(
            epochs=epochs,
            batch=trial.suggest_categorical("batch", [32, 64, 128]),
            lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            patience=patience,
            grad_clip=trial.suggest_float("grad_clip", 0.5, 2.0),
            label_smoothing=trial.suggest_float("label_smoothing", 0.0, 0.1),
            val_split=val_split,
            seed=seed,
        )

        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            model, best_score, best_epoch = train_main(cfg)  # <- ver siguiente sección
            mlflow.log_metric("trial_best_val_f1_macro", float(best_score))
            mlflow.log_param("trial_best_epoch", int(best_epoch))
            return float(best_score)

    return objective



def main(
    n_trials=20,
    study_name="ocr-hpo",
    epochs=60,
    patience=8,
    val_split=0.2,
    seed=42,
):
    """
    Runs Optuna HPO and then trains ONE final model using the best params.
    Returns:
      study, best_model
    """
    with mlflow.start_run(run_name="hpo-optuna", nested=True):
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("study_name", study_name)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("patience", patience)
        mlflow.log_param("val_split", val_split)
        mlflow.log_param("seed", seed)

        study = optuna.create_study(direction="maximize", study_name=study_name)
        objective = build_objective(
            epochs=epochs,
            patience=patience,
            val_split=val_split,
            seed=seed,
        )
        study.optimize(objective, n_trials=n_trials)

        # Log best summary
        mlflow.log_param("best_trial", study.best_trial.number)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_value", float(study.best_value))

        print("Best:", study.best_value, study.best_params)

        # -------------------------------------------------------
        # FINAL TRAINING: train once with best params -> best_model
        # -------------------------------------------------------
        best_cfg = TrainConfig(
            epochs=epochs,
            batch=study.best_params["batch"],
            lr=study.best_params["lr"],
            weight_decay=study.best_params["weight_decay"],
            patience=patience,
            grad_clip=study.best_params["grad_clip"],
            label_smoothing=study.best_params["label_smoothing"],
            val_split=val_split,
            seed=seed,
        )

        # This creates another nested run under the HPO run
        with mlflow.start_run(run_name="best-model-train", nested=True):
            best_model, best_score, best_epoch = train_main(best_cfg)
            mlflow.log_metric("best_model_val_f1_macro", float(best_score))
            mlflow.log_param("best_model_best_epoch", int(best_epoch))

        #return study, best_model
        return best_model



if __name__ == "__main__":
    # Choose one:
    # 1) Single training run:
    #main(epochs=40, batch=64, lr=1e-3, weight_decay=1e-4)

    # 2) Hyperparameter optimization:
    main_optuna(n_trials=20, study_name="ocr-hpo")
