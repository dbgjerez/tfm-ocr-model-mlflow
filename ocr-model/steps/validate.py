import time
import json
import mlflow
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
)
from mlflow.tracking import MlflowClient

from steps.data import CharsDataset, transform_ocr, DATASET_DIR, CSV_PATH


# -----------------------------------------------------------
# Tiempo de inferencia (latencia)
# -----------------------------------------------------------
def infer_latency(model, device="cpu", repetitions=50):
    dummy = torch.randn(1, 1, 32, 32).to(device)

    # Warmup
    for _ in range(10):
        _ = model(dummy)

    # Medición
    times = []
    for _ in range(repetitions):
        start = time.time()
        _ = model(dummy)
        times.append(time.time() - start)

    return {
        "latency_mean_ms": float(np.mean(times) * 1000),
        "latency_p95_ms": float(np.percentile(times, 95) * 1000),
        "latency_p99_ms": float(np.percentile(times, 99) * 1000),
    }



# -----------------------------------------------------------
# VALIDACIÓN COMPLETA
# -----------------------------------------------------------
def main(model):
    # Crear child run
    with mlflow.start_run(run_name="validate", nested=True):

        # ---------------------------
        # Cargar modelo entrenado
        # ---------------------------
        model.eval()

        # ---------------------------
        # Preparar datos
        # ---------------------------
        df = pd.read_csv(CSV_PATH)
        dataset = CharsDataset(df, root_dir=DATASET_DIR, transform=transform_ocr)

        imgs, labels = [], []
        for img, label in dataset:
            imgs.append(img.numpy())
            labels.append(label)

        imgs = torch.tensor(np.array(imgs))
        labels = np.array(labels)

        # Predicciones
        with torch.no_grad():
            preds = model(imgs).argmax(1).numpy()

        # ---------------------------
        # MÉTRICAS DE DESEMPEÑO
        # ---------------------------
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")
        rec = recall_score(labels, preds, average="macro")
        prec = precision_score(labels, preds, average="macro")

        mlflow.log_metric("val_accuracy", acc)
        mlflow.log_metric("val_f1_macro", f1)
        mlflow.log_metric("val_recall_macro", rec)
        mlflow.log_metric("val_precision_macro", prec)

        print(f"[VALIDACIÓN] Accuracy = {acc:.4f}")

        # Matriz de confusión
        cm = confusion_matrix(labels, preds)
        Path("artifacts/validate").mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(12, 9))
        sns.heatmap(cm, cmap="Blues")
        plt.title("Confusion Matrix")
        cm_path = "artifacts/validate/confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)

        # ---------------------------
        # VALIDACIÓN OPERATIVA
        # ---------------------------
        device = "cpu"
        model.to(device)

        latency = infer_latency(model)

        for k, v in latency.items():
            mlflow.log_metric(k, v)

        # Tamaño del modelo
        temp_path = "artifacts/validate/temp_model.pt"
        torch.save(model.state_dict(), temp_path)
        model_size = Path(temp_path).stat().st_size / (1024 * 1024)
        mlflow.log_metric("model_size_mb", model_size)

        # ---------------------------
        # Reporte JSON
        # ---------------------------
        report = {
            "accuracy": acc,
            "f1_macro": f1,
            "recall_macro": rec,
            "precision_macro": prec,
            "latency": latency,
            "model_size_mb": model_size,
        }

        json_path = "artifacts/validate/report.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=4)
        mlflow.log_artifact(json_path)

        print("[VALIDACIÓN] Etapa completada correctamente.")


if __name__ == "__main__":
    main()
