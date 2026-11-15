# E-commerce Customer Churn Prediction System

## Introduction
this project implements an end_to_end project Machine Learning Pipeline to predict customer churn in an e-commerce enviroment

## Source
**Data Source:** Kaggle

## Tools
* **Modeling :** "OneHotEncoder", "StandardScaler", "SimpleImputer", "FeatureSelection", "SMOTE", "LightGBM"
* **Versioning :** "MLFlow" (Experiment traking & Regitry Model)
* **API Serving :** "FastAPI"
* **Validation :** "Pydantic"
* **Monitoring :** "Evidently AI"
* **Testing :** "pytest"

## Key Features
1. **Pipeline Integrity :** uses impipeline to prevent data leakage during smote oversampling
2. **MLOPS Lifecycle :** full integration with MLFlow for expriment reproducibility and versioning
3. **Drift Prediction :** Data monitoring system for examining changes in data distribution
4. **Validation Data :** Data structure validation before reaching the model
5. **Model Serving :** Serving a Model with FastAPI for Production Use
6. **Deployment :** Containerizing (Dockerizing) the application for model deployment on servers.

## Project Structure
1. **api_model.py :**            # FastAPI implementation & inference engine
2. **model_training.py :**       # MLflow-integrated training script
3. **pipeline_lgbm.py :**        # Preprocessing & Pipeline definition
4. **validation.py :**           # Pydantic schemas for input validation
5. **data_monitoring.py :**      # Evidently-based Drift Detection
6. **test_api.py :**             # Integration tests
7. **requirements.txt :**        # Project dependencies
8. **Files/ :**                  # Artifacts, Reference data, and Feature names