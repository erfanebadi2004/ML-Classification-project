import logging
import joblib
import pandas as pd
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, BackgroundTasks
from validation import BatchPredictionRequest
from pathlib import Path
from data_monitoring import check_data_drift

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler("app_errors.log"), logging.StreamHandler()])

logger = logging.getLogger("Lgbm Mode Api")

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURE_PATH = BASE_DIR / "src" / "./Files" / "features_name.pkl"
REFERENCE_PATH = BASE_DIR / "src" / "./Files" / "reference.csv"
MODEL_PATH = BASE_DIR / "src" / "./LgbmClassification_folder"

try:
    df_reference = pd.read_csv(REFERENCE_PATH)
    logger.info("Reference data loaded successfully")

    with open(FEATURE_PATH, "rb") as features:
        features_name = joblib.load(features)

except Exception as e:
        logger.error(f"Critical error: Reference data failed to load. {REFERENCE_PATH}: {str(e)}")

try:
    lgbm_model = mlflow.pyfunc.load_model(MODEL_PATH)
    logger.info(f"Model loaded successfully: {MODEL_PATH}")

except Exception as error:
    logger.exception(f"Model loading failed{str(error)}")

app = FastAPI(title="LGBMClassificationModel_API")

@app.post("/predict")
def predict_lgbm_classification(request: BatchPredictionRequest, background_tasks: BackgroundTasks):

    if lgbm_model is None:
        logger.critical("The model failed to load!")
        raise HTTPException(status_code=500, detail="Server configuration error: Model not found.")

    if not request.samples:
        logger.warning("Empty samples received.")
        raise HTTPException(status_code=400, detail="No samples were found.")

    try:
        df_samples = pd.DataFrame([s.model_dump() for s in request.samples])

        df = df_samples[features_name]

        predict_model = lgbm_model.predict(df)
        result = predict_model.tolist()

        background_tasks.add_task(check_data_drift, df_reference, df)

        return result

    except Exception as E:
        logger.error(f"Error during the prediction process: {str(E)}")
        raise HTTPException(status_code=500,detail=f"Prediction error: {str(E)}" )