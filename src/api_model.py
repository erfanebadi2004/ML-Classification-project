import logging
import joblib
import pandas as pd
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, BackgroundTasks
from validation import BatchPredictionRequest
from pathlib import Path
from data_monitoring import check_data_drift

# تنظیم سیستم logging برای ذخیره و نمایش گزارش ها در زمان اجرا برنامه در محیط سرور
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler("app_errors.log"), logging.StreamHandler()])

logger = logging.getLogger("Lgbm Mode Api")

# تعیین مسیر فایل های مدل و داده های مرجع و نام ستون ها
BASE_DIR = Path(__file__).resolve().parent.parent
FEATURE_PATH = BASE_DIR / "src" / "./Files" / "features_name.pkl"
REFERENCE_PATH = BASE_DIR / "src" / "./Files" / "reference.csv"
MODEL_PATH = BASE_DIR / "src" / "./LgbmClassification_folder"

# بارگزاری داده های مرجع reference و نام ستون ها قبل از شروع کار API برای جلوگیری از خواندن مجدد فایل ها در محیط سرور
try:
    df_reference = pd.read_csv(REFERENCE_PATH)
    logger.info("Reference data loaded successfully")

    with open(FEATURE_PATH, "rb") as features:
        features_name = joblib.load(features)

except Exception as e:
        logger.error(f"Critical error: Reference data failed to load. {REFERENCE_PATH}: {str(e)}")

# بارگزاری مدل light gbm از MLFlow
try:
    lgbm_model = mlflow.pyfunc.load_model(MODEL_PATH)
    logger.info(f"Model loaded successfully: {MODEL_PATH}")

except Exception as error:
    logger.exception(f"Model loading failed{str(error)}")

# راه اندازی اپلیکیشن FastAPI
app = FastAPI(title="LGBMClassificationModel_API")

# تعریف مسیر endpoint برای دریافت درخواست های پیش بینی
@app.post("/predict")
def predict_lgbm_classification(request: BatchPredictionRequest, background_tasks: BackgroundTasks):

    # بررسی مدل بارگزاری شده برای زمانی که مدل به درستی بارگزاری نشده خطا 500 برمیگرداند
    if lgbm_model is None:
        logger.critical("The model failed to load!")
        raise HTTPException(status_code=500, detail="Server configuration error: Model not found.")

    # زمانی که داده ای در درخواست ها به سمت سرور وجود نداشته باشد خطا 400 برمیگرداند
    if not request.samples:
        logger.warning("Empty samples received.")
        raise HTTPException(status_code=400, detail="No samples were found.")

    try:
        # تبدیل داده های دریافتی که توسط pydantic اعتبار سنجی شده به DataFrame
        df_samples = pd.DataFrame([s.model_dump() for s in request.samples])

        # انتخاب ستون های مورد نیاز و مرتب سازی بر اساس ترتیب نام ستون های feature_name
        df = df_samples[features_name]

        # انجام فرایند پیش بینی توسط مدل
        predict_model = lgbm_model.predict(df)
        result = predict_model.tolist()

        # ثبت وظایف در پس زمینه برای بررسی انحراف داده drift
        background_tasks.add_task(check_data_drift, df_reference, df)

        return result

    except Exception as E:
        logger.error(f"Error during the prediction process: {str(E)}")
        raise HTTPException(status_code=500,detail=f"Prediction error: {str(E)}" )