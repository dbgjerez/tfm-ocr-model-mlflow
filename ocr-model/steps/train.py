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
from sklearn.metrics import confusion_matrix

from steps.data import CharsDataset, transform_ocr, NUM_CLASSES, DATASET_DIR, CSV_PATH


# -----------------------------------------------------------
# MODELO CNN SIMPLE
# -----------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------------------------------------
# PREPARACIÓN DEL DATASET
# -----------------------------------------------------------
def preparar_dataset(csv_path, root=DATASET_DIR, val_split=0.2):
    df = pd.read_csv(csv_path)
    dataset = CharsDataset(df, root_dir=root, transform=transform_ocr)

    val_len = int(len(dataset) * val_split)
    train_len = len(dataset) - val_len

    return random_split(dataset, [train_len, val_len])


# -----------------------------------------------------------
# ENTRENAMIENTO
# -----------------------------------------------------------
def train_one_epoch(model, loader, device, opt, crit):
    model.train()
    total_loss, correct = 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        opt.zero_grad()
        out = model(imgs)
        loss = crit(out, labels)
        loss.backward()
        opt.step()

        total_loss += loss.item()
        correct += (out.argmax(1) == labels).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)


def validar(model, loader, device, crit):
    model.eval()
    total_loss, correct = 0, 0
    preds, labels_all = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = crit(out, labels)

            total_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item()

            preds.extend(out.argmax(1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    return total_loss / len(loader), correct / len(loader.dataset), labels_all, preds


def guardar_confusion(labels, preds, path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, cmap="Blues")
    plt.title("Matriz de Confusión")
    plt.tight_layout()
    plt.savefig(path)
    return path


# -----------------------------------------------------------
# MAIN – CHILD RUN
# -----------------------------------------------------------
def main(csv="chars74k_labels.csv", epochs=3, batch=64):

    with mlflow.start_run(run_name="train", nested=True):

        out_dir = Path("artifacts/modelado")
        out_dir.mkdir(parents=True, exist_ok=True)

        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch)

        # Dataset
        train_ds, val_ds = preparar_dataset(CSV_PATH)
        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch)

        # Modelo
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SimpleCNN().to(device)
        crit = nn.CrossEntropyLoss()
        opt = optim.Adam(model.parameters(), lr=1e-3)

        # Entrenamiento
        for ep in range(epochs):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, device, opt, crit)
            val_loss, val_acc, labels, preds = validar(model, val_loader, device, crit)

            mlflow.log_metric("train_loss", tr_loss, step=ep)
            mlflow.log_metric("train_acc", tr_acc, step=ep)
            mlflow.log_metric("val_loss", val_loss, step=ep)
            mlflow.log_metric("val_acc", val_acc, step=ep)

            print(f"[Epoch {ep}] Train acc={tr_acc:.3f} | Val acc={val_acc:.3f}")

        # Matriz de confusión
        cm_path = guardar_confusion(labels, preds, out_dir / "confusion_matrix.png")
        mlflow.log_artifact(str(cm_path))

        # -----------------------------------------------------
        # INPUT EXAMPLE Y FIRMA (CORRECTO PARA MLFLOW)
        # -----------------------------------------------------
        input_example_tensor = torch.zeros((1, 1, 32, 32))      # <- tensor dummy
        input_example = input_example_tensor.numpy()            # <- MLflow require numpy!

        pred_example = model(input_example_tensor).detach().cpu().numpy()

        signature_cnn = infer_signature(input_example, pred_example)

        # Registrar modelo original (CNN)
        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="ocr-model",
            signature=signature_cnn,
            input_example=input_example
        )

        print("[MODELADO] Etapa completada correctamente")

        return model   # <- se envía a deploy.py



if __name__ == "__main__":
    main()
