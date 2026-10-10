import logging
import pandas as pd
import mlflow.sklearn
from pipeline_lightgbm import pipeline_lgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score

# تنظیمات مربوط به logging برای ذخیره گزارش ها و نمایش در کنسول
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler("file_info.log"), logging.StreamHandler()])

# ایجاد شی logger برای ثبت وقایع و فرایند آموزش مدل
logger = logging.getLogger("Training Model")

# بارگزاری داده های مربوط به ریزش مشتریان
df_customers = pd.read_csv("../data/ecommerce_churn_data.csv")

# تبدیل ستون target به مقادیر عددی 0 و 1
y = df_customers["Churn"].map({0: 0, 1: 1}).values.reshape(-1)

# حذف ستون های غیر ضروری و ستون هدف target از ویژگی های X
X = df_customers.drop(columns=df_customers[["CustomerID", "Complain", "Churn"]])

# تعریف لیست نام ویژگی های عددی و دسته ای برای استفاده در pipeline
numeric_list = ["Age", "Tenure", "WarehouseToHome", "HoursSpentOnApp", "OrderCount", "DaySinceLastOrder", "CashbackAmount"]
category_list = ["Gender", "PreferredLoginDevice", "CityTier", "NumberOfDevicesRegistered", "SatisfactionScore", "NumberOfAddress"]

# تقسیم داده ها به دو قسمت train و test با نسبت 80 درصد برای آموزش و 20 درصد برای ارزیابی
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1234)

# ذخیره داده آموزشی به عنوان فایل مرجع و داده تست برای ارزیابی و بازتولید
X_train.to_csv('reference.csv', index=False)
X_test.to_csv('prediction_test.csv', index=False)

# ساخت شی pipeline_lgbm به همراه قابلیت های scaling , encoding , missing , feature_selection
pipeline = pipeline_lgbm(numeric_features=numeric_list, category_features=category_list)

# شروع فرایند آموزش مدل
logger.info("Starting the model training process")
pipeline.fit(X_train, y_train)

# شروع مرحله ارزیابی مدل با داده های تست
logger.info("Starting the evaluation phase using test data.")
y_predict = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

# محاسبه معیار های ارزیابی
acc = accuracy_score(y_test, y_predict)
precision = precision_score(y_test, y_predict)
recall = recall_score(y_test, y_predict)
auc = roc_auc_score(y_test, y_prob)
logger.info(f"Accuracy Score : {acc}, Precision Score : {precision}, Recall Score : {recall}, AUC Score : {auc}")

skops_trusted_types = [
    'imblearn.over_sampling._smote.base.SMOTE',
    'imblearn.pipeline.Pipeline',
    'sklearn.metrics._dist_metrics.EuclideanDistance64',
    'sklearn.neighbors._kd_tree.KDTree',
    'collections.OrderedDict',
    'lightgbm.basic.Booster',
    'lightgbm.sklearn.LGBMClassifier',
    'numpy.dtype'
]

# فعال سازی autolog برای ثبت خودکار پارامتر های lightgbm در سرور MLFlow
mlflow.lightgbm.autolog()

# تعریف آدرس سرور MLFlow برای اتصال به سرور
tracking_uri = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(tracking_uri)

# تنظیم آزمایش
mlflow.set_experiment("EcommerceCustomersChurn")

# شروع یک RUN جدید در MLFlow برای ثبت نتایج مدل
with mlflow.start_run():
    # تعریف هایپر پارامتر ها و پارامتر های مدل
    params = {"max_depth": 5, "min_data_in_leaf": 3, "min_gain_to_split": 0.8}

    # ثبت پارامتر ها به صورت دستی
    mlflow.log_params(params)
    logger.info("Registering model parameters")

    # ثبت معیار های ارزیابی در MLFlow
    mlflow.log_metric("acc", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("auc", auc)
    logger.info("Registering model evaluation metrics")

    # ثبت و ذخیره مدل در mlflow model registery برای مدیریت نسخه های مدل
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        skops_trusted_types=skops_trusted_types,
        name="LightGBM",
        registered_model_name="LightgbmModel",
    )
    logger.info("Registering and saving the trained model")

    print(f"✅ Success! Model logged via Joblib. AUC: {auc:.4f}")

# تعیین آدرس آخرین نسخه مدل در MLFlow برای دانلود
model_uri = "models:/LightgbmModel/latest"
download_path = "./LgbmClassification_folder"

# دانلود artifact های مدل از سرور MLFlow در سیستم محلی
mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=download_path)
logger.info("Download model artifact files from MLflow.")