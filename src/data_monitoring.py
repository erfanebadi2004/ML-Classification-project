import logging
from evidently import Report
from evidently.presets import DataDriftPreset

# تعریف متد check_data_drift برای مقایسه و بررسی تغییر در توزیع داده های مرجع و داده های جدید
def check_data_drift(reference=None ,current=None):

    # ایجاد یک گزارش برای بررسی data drift
    report = Report(metrics=[DataDriftPreset()])

    # اجرا فرایند مقایسه توزیع داده های مرجع و داده های جدید
    result = report.run(reference_data=reference, current_data=current)

    # تبدیل نتایج گزارش به دیکشنری برای دسترسی آسان به اعداد
    # اعداد به دست آمده توصیف کننده این موضوع است که چند درصد از ستون ها دچار تغییر شده اند
    result_drift = result.dict()

    # تصمیم گیری بر اساس آستانه threshold , 50 درصدی
    if result_drift["metrics"][0]["value"]["share"] > 0.5:
        # اگر بیش از 50 درصد ویژگی ها تغییر کرده باشند هشدار بحرانی صادر خواهد شد
        print(f"⚠️ DATA DRIFT DETECTED! Score: {result_drift["metrics"][0]["value"]["share"]}")
    else:
        # در غیر این صورت اطلاع رسانی میکند که تغییری در داده ها وجود ندارد
        print(f"No change was found in the distribution of inputs, score : {result_drift["metrics"][0]["value"]["share"]}")
