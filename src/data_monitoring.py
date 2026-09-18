from evidently import Report
from evidently.presets import DataDriftPreset

def check_data_drift(reference=None ,current=None):

    report = Report(metrics=[DataDriftPreset()])

    result = report.run(reference_data=reference, current_data=current)

    result_drift = result.dict()

    if result_drift["metrics"][0]["value"]["share"] > 0.5:
        print(f"⚠️ DATA DRIFT DETECTED! Score: {result_drift["metrics"][0]["value"]["share"]}")
    else:
        print(f"No change was found in the distribution of inputs, score : {result_drift["metrics"][0]["value"]["share"]}")
