import pytest
import pandas as pd
from fastapi.testclient import TestClient
from api_model import app

client = TestClient(app)

@pytest.fixture
def sample_payload():
    prediction_test = pd.read_csv("Files/prediction_test.csv")

    data_list = prediction_test.to_dict(orient='records')

    return {"samples": data_list}


def test_prediction(sample_payload):
    response = client.post("/predict", json=sample_payload)

    result_response = response.json()

    print("\n--- Model Predictions ---")
    print(result_response)
    print("===== ===== ===== ===== =====")

    assert response.status_code == 200