import mlflow

from datetime import datetime
from steps.data import main as data_main
from steps.train import main as train_main
from steps.validate import main as validate_main
from steps.deploy import main as deploy_main

mlflow.set_experiment("tfm-ocr-chars")

def execution_name():
    now = datetime.now().strftime("%Y%m%d")
    return f"pipeline-{now}"

with mlflow.start_run(run_name=execution_name()) as parent_run:

    print("==== ETAPA 1: DATOS ====")
    data_main()

    print("==== ETAPA 2: ENTRENAMIENTO ====")
    model = train_main()     

    print("==== ETAPA 3: VALIDACIÓN ====")
    validate_main(model)     

    print("==== ETAPA 4: DESPLIEGUE ====")
    deploy_main(model)

print("Pipeline terminado.")
