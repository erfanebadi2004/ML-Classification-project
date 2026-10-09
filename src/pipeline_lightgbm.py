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
    # ساخت (pipeline) برای ویژگی های عددی
    # اول مقادیر خالی رو با استفاده از تکنیک (SimpleImputer) با استراتژی (median) پر میکنیم
    # بعد ویژگی های عددی رو با استفاده از تکنیک (StandardScaler) در یک مقیاس یکساس تنظیم میکنم تا تمام ویژگی ها در یک مقیاس قرار داشته باشند
    numeric_transformer = Pipeline(steps=[
        ("MissingValues", SimpleImputer(strategy="median")),
        ("StandardScaler", StandardScaler())
    ])

    # ساخت (pipeline) برای ویژگی های دسته ای
    # اول مقادیر خالی را با همان تکنیک (SimpleImputer) با استراتژی (most frequent) پر میکنیم
    # بعد ویژگی های دسته ای رو با استفاده از تکنیک (OneHotEncoder) به بردار های 0 و 1 تبدیل میکنیم
    category_transformer = Pipeline(steps=[
        ("MissingValues", SimpleImputer(strategy="most_frequent")),
        ("OneHotEncoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # انجام چند نوع تبدیل مختلف مثل missing handeling و encoding و scaling به صورت همزمان
    # و تطبیق ستون ها با (transformer) های مرتبط
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("category", category_transformer, category_features)
    ])

    # ساخت خط لوله نهایی با استفاده از (ImbPipeline) برای جلوگیری از نفوذ داده های غیر واقعی به داده های تست
    # با هدف جلوگیری از نشت داده (Date Leakage)
    full_pipeline = ImbPipeline(steps=[
        # بخش اول : پیش پرداش داده ها
        ("Preprocess", preprocessor),
        # بخش دوم : انتخاب ویژگی ها
        # استفاده از LogisticRegression برای انتخاب ویژگی های مهمتر
        ("FeatureSelection", SelectFromModel(estimator=LogisticRegression())),
        # بخش سوم : مدیریت عدم توازن کلاس ها
        # تولید داده های غیر واقعی برای کلاس های اقلیت (عدم استفاده در داده های تست)
        ("Over_Sampling", SMOTE(random_state=1234)),
        # بخش چهارم : مدل سازی با استفاده از الگوریتم LightGBM
        ("Classifier", LGBMClassifier(max_depth=9, min_data_in_leaf=5, min_gain_to_split=0.7, random_state=1234)),
    ])

    return full_pipeline