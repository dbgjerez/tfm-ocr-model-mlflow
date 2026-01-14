import mlflow
import mlflow.pyfunc
import torch
import base64
import io
import numpy as np
from PIL import Image
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.types.schema import Schema, ColSpec

from steps.data import transform_ocr, label_to_char


# ============================================================
# MODELO REST PYFUNC
# ============================================================
class OCRPyFunc(mlflow.pyfunc.PythonModel):

    def __init__(self, pytorch_model):
        self.model = pytorch_model.eval()

    def predict(self, context, model_input):
        # MLflow REST → DataFrame
        if hasattr(model_input, "iloc"):  # pandas DataFrame
            b64 = model_input["image_base64"].iloc[0]
        elif isinstance(model_input, dict):  # dict recibido directamente
            b64 = model_input["image_base64"]
        else:
            raise ValueError(f"Tipo de entrada no soportado: {type(model_input)}")

        img_bytes = base64.b64decode(b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        tensor = transform_ocr(img).unsqueeze(0)

        with torch.no_grad():
            out = self.model(tensor)
            pred_idx = int(out.argmax(1).item())

        return {"prediction": label_to_char[pred_idx]}



# ============================================================
# MAIN – REGISTRO DEL MODELO REST EN MLFLOW
# ============================================================
def main(model):

    with mlflow.start_run(run_name="deploy", nested=True):

        print("[DEPLOY] Preparando modelo REST PyFunc...")

        # ---------------------------------------------------
        # 1) Crear input_example REAL con base64 válido
        # ---------------------------------------------------
        dummy_img = Image.new("L", (32, 32), color=0)
        buf = io.BytesIO()
        dummy_img.save(buf, format="PNG")
        dummy_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        input_example = {"image_base64": dummy_b64}

        # ---------------------------------------------------
        # 2) Firma del modelo REST
        # ---------------------------------------------------
        input_schema = Schema([ColSpec("string", "image_base64")])
        output_schema = Schema([ColSpec("string", "prediction")])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        # ---------------------------------------------------
        # 3) REGISTRAR MODELO REST
        # ---------------------------------------------------
        result = mlflow.pyfunc.log_model(
            name="ocr-model-rest",
            python_model=OCRPyFunc(model),
            signature=signature,
            #input_example=input_example,
            registered_model_name="ocr_model_rest",
        )

        print(f"[DEPLOY] Modelo REST registrado correctamente.")
        print(f"[DEPLOY] Versión registrada: {result.registered_model_version}")

        return result


if __name__ == "__main__":
    raise RuntimeError("Este script debe ejecutarse desde pipeline.py")
