# 🧠 OCR Model with MLflow (TFM)

This repository contains the code and artifacts for a Master’s Thesis (TFM) project focused on building an **Optical Character Recognition (OCR)** pipeline with **MLflow** for experiment tracking and reproducibility.

The main objectives are:

- Train and evaluate OCR models
- Track experiments (parameters, metrics, artifacts) with MLflow
- Keep the workflow reproducible and easy to iterate on

---

## ✨ What’s inside

- OCR training and evaluation code (PyTorch-based)
- Notebooks for experimentation and validation
- Inference utilities for running predictions
- Trained model artifacts included in the repository (for reference)

> Note: The repository may contain additional files (test images, CSVs, and auxiliary models) used during experimentation.

---

## 📁 Project layout (high level)

Typical structure you will find in this repository:

- `ocr-model/` – core training / evaluation code
- `ocr.ipynb` – training notebook (experimentation + MLflow logging)
- `ocr-valid.ipynb` – validation / evaluation notebook
- `reader.py` – lightweight inference helper (load model + predict)

---

## ⚙️ Requirements

- Python **3.9+** recommended (3.8+ should also work in most cases)
- Dependencies listed in `requirements.txt` (if present)

Install dependencies:

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt` yet, you can generate one from your environment:

```bash
pip freeze > requirements.txt
```

---

## 🚀 Running MLflow

Start the MLflow UI locally:

```bash
mlflow ui
```

Default address:

- `http://127.0.0.1:5000`

MLflow will track runs in the local `mlruns/` directory unless configured otherwise.

---

## 🏋️ Training

Open and run the training notebook:

- `ocr.ipynb`

This typically performs:

- data loading / preprocessing
- model training
- metric computation
- MLflow logging (params, metrics, artifacts, model)

---

## 📊 Validation

Use the validation notebook:

- `ocr-valid.ipynb`

This typically includes:

- evaluation metrics
- qualitative inspection (predictions on sample images)
- confusion matrix / per-class analysis (if applicable)

---

## 🔎 Inference

Example usage (adjust names to match your code if needed):

```python
from reader import OCRReader

ocr = OCRReader(model_path="ocr_model.pt")
text = ocr.predict("path/to/image.jpg")
print(text)
```

---

## 🧪 Reproducibility notes

To improve determinism between runs:

- Fix random seeds (Python / NumPy / PyTorch)
- Control data splits
- Disable non-deterministic ops where possible (PyTorch flags)
- Track:
  - seed value
  - dataset version / hash
  - git commit
  - training parameters

These items can be logged in MLflow to make results easier to validate and compare.

---

## 📜 License

This project is licensed under the **MIT License**. See `LICENSE`.
