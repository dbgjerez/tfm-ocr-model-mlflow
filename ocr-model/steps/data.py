import os
import json
import pandas as pd
import string
import sys
from pathlib import Path
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset


# ============================================================
# CONFIGURACIÓN
# ============================================================
DATASET_DIR = "./English/Fnt"
CSV_PATH = "./chars74k_labels.csv"

digits = "0123456789"
uppercase = string.ascii_uppercase

char_to_label = {d: i for i, d in enumerate(digits)}
for i, ch in enumerate(uppercase, start=10):
    char_to_label[ch] = i

label_to_char = {v: k for k, v in char_to_label.items()}
NUM_CLASSES = len(char_to_label)

transform_ocr = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5]),
])


class CharsDataset(Dataset):
    def __init__(self, df, root_dir=DATASET_DIR, transform=None):
        self.df = df.reset_index(drop=True)
        self.root = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root, row["path"])
        label_idx = char_to_label[row["label"]]

        img = Image.open(img_path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, label_idx


# ============================================================
# FUNCIONES DE DATA
# ============================================================
def recoleccion_datos(dataset_dir: str) -> pd.DataFrame:
    rows = []
    samples = sorted(os.listdir(dataset_dir))

    for sample in samples:
        if not sample.startswith("Sample"):
            continue

        try:
            sample_num = int(sample.replace("Sample", ""))
        except ValueError:
            continue

        folder = os.path.join(dataset_dir, sample)
        for fname in os.listdir(folder):
            if fname.lower().endswith(".png"):
                rows.append({
                    "path": os.path.join(sample, fname),
                    "sample": sample,
                    "sample_num": sample_num
                })

    return pd.DataFrame(rows)


def limpieza_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def map_label(sample_num: int):
        if 1 <= sample_num <= 10:
            return str(sample_num - 1)
        elif 11 <= sample_num <= 36:
            return chr(ord("A") + (sample_num - 11))
        else:
            return None  # minúsculas u otros símbolos

    df["label"] = df["sample_num"].apply(map_label)

    # filtrado final
    df = df[df["label"].notnull()].reset_index(drop=True)

    return df



def transformacion_y_enriquecimiento(df: pd.DataFrame, out_dir: Path):
    mapping_path = out_dir / "mapping.json"
    mapping_path.write_text(json.dumps({
        "char_to_label": char_to_label,
        "label_to_char": label_to_char
    }, indent=4))
    return df, mapping_path


def generar_eda(df: pd.DataFrame, eda_dir: Path):
    plt.figure(figsize=(12, 4))
    sns.countplot(data=df, x="label", order=sorted(df["label"].unique()))
    plt.title("Distribución de clases en Chars74K")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plot_path = eda_dir / "class_distribution.png"
    plt.savefig(plot_path)
    return plot_path


def versionado_datos(df: pd.DataFrame, out_dir: Path):
    version_file = out_dir / "version.json"
    version_file.write_text(json.dumps({
        "dataset_version": "v1",
        "num_samples": len(df),
        "num_classes": NUM_CLASSES
    }, indent=4))
    return version_file


# ============================================================
# MAIN (CHILD RUN)
# ============================================================
def main():

    # CHILD RUN
    with mlflow.start_run(run_name="datos", nested=True):

        out_dir = Path("artifacts/datos")
        eda_dir = out_dir / "eda"
        out_dir.mkdir(parents=True, exist_ok=True)
        eda_dir.mkdir(parents=True, exist_ok=True)

        mlflow.log_param("dataset_dir", DATASET_DIR)

        # 1. Recolección
        df_raw = recoleccion_datos(DATASET_DIR)
        mlflow.log_metric("raw_samples", len(df_raw))

        # 2. Limpieza
        df_clean = limpieza_datos(df_raw)
        mlflow.log_metric("clean_samples", len(df_clean))

        # 3. Transformación
        df_transformed, mapping_path = transformacion_y_enriquecimiento(df_clean, out_dir)
        mlflow.log_artifact(str(mapping_path))

        # 4. EDA
        plot_path = generar_eda(df_transformed, eda_dir)
        mlflow.log_artifact(str(plot_path))

        # 5. Versionado
        version_file = versionado_datos(df_transformed, out_dir)
        mlflow.log_artifact(str(version_file))

        # CSV final
        df_transformed.to_csv(CSV_PATH, index=False)
        mlflow.log_artifact(CSV_PATH)

        print("[DATOS] Subetapas completadas correctamente")


if __name__ == "__main__":
    main()
