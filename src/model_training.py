import pandas as pd
import mlflow.sklearn
from pipeline_lightgbm import pipeline_lgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score

df_customers = pd.read_csv("../data/ecommerce_churn_data.csv")

y = df_customers["Churn"].map({0: 0, 1: 1}).values.reshape(-1)

X = df_customers.drop(columns=df_customers[["CustomerID", "Complain", "Churn"]])

numeric_list = ["Age", "Tenure", "WarehouseToHome", "HoursSpentOnApp", "OrderCount", "DaySinceLastOrder", "CashbackAmount"]
category_list = ["Gender", "PreferredLoginDevice", "CityTier", "NumberOfDevicesRegistered", "SatisfactionScore", "NumberOfAddress"]

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1234)

X_train.to_csv('reference.csv', index=False)
X_test.to_csv('prediction_test.csv', index=False)

pipeline = pipeline_lgbm(numeric_features=numeric_list, category_features=category_list)

pipeline.fit(X_train, y_train)

y_predict = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_predict)
precision = precision_score(y_test, y_predict)
recall = recall_score(y_test, y_predict)
auc = roc_auc_score(y_test, y_prob)

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


mlflow.lightgbm.autolog()
tracking_uri = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("EcommerceCustomersChurn")

with mlflow.start_run():
    params = {"max_depth": 5, "min_data_in_leaf": 3, "min_gain_to_split": 0.8}

    mlflow.log_params(params)
    mlflow.log_metric("acc", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("auc", auc)

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        skops_trusted_types=skops_trusted_types,
        name="LightGBM",
        registered_model_name="LightgbmModel",
    )

    print(f"✅ Success! Model logged via Joblib. AUC: {auc:.4f}")

model_uri = "models:/LightgbmModel/latest"
download_path = "./LgbmClassification_folder"

mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=download_path)