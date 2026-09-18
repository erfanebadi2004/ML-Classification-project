from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

def pipeline_lgbm(numeric_features=None, category_features=None):
    numeric_transformer = Pipeline(steps=[
        ("MissingValues", SimpleImputer(strategy="median")),
        ("StandardScaler", StandardScaler())
    ])

    category_transformer = Pipeline(steps=[
        ("MissingValues", SimpleImputer(strategy="most_frequent")),
        ("OneHotEncoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("category", category_transformer, category_features)
    ])

    full_pipeline = ImbPipeline(steps=[
        ("Preprocess", preprocessor),
        ("FeatureSelection", SelectFromModel(estimator=LogisticRegression())),
        ("Over_Sampling", SMOTE(random_state=1234)),
        ("Classifier", LGBMClassifier(max_depth=9, min_data_in_leaf=5, min_gain_to_split=0.7, random_state=1234)),
    ])

    return full_pipeline