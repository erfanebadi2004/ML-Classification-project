import pytest
from fastapi.testclient import TestClient
from api_model import app

client = TestClient(app)

@pytest.fixture
def valid_sample_input():
    return {"samples": [
        {"Age": 29, "Gender": "Male", "Tenure": 5.0, "PreferredLoginDevice": "Desktop", "CityTier": 1,
         "WarehouseToHome": 7.2,
        "HoursSpentOnApp": 2.5, "NumberOfDevicesRegistered": 2, "SatisfactionScore": 3, "NumberOfAddress": 1,
        "OrderCount": 8, "DaySinceLastOrder": 15, "CashbackAmount": 45.2},
        {"Age": 45, "Gender": "Female", "Tenure": 36.0, "PreferredLoginDevice": "Mobile", "CityTier": 3,
         "WarehouseToHome": 20.1,
         "HoursSpentOnApp": 4.8, "NumberOfDevicesRegistered": 4, "SatisfactionScore": 5, "NumberOfAddress": 3,
         "OrderCount": 50, "DaySinceLastOrder": 2, "CashbackAmount": 320.5},
        {"Age": 22, "Gender": "Male", "Tenure": 2.0, "PreferredLoginDevice": "Mobile", "CityTier": 1,
         "WarehouseToHome": 5.0,
         "HoursSpentOnApp": 1.8, "NumberOfDevicesRegistered": 1, "SatisfactionScore": 2, "NumberOfAddress": 1,
         "OrderCount": 3, "DaySinceLastOrder": 25, "CashbackAmount": 10.1},
        {"Age": 39, "Gender": "Female", "Tenure": 24.0, "PreferredLoginDevice": "Desktop", "CityTier": 2,
         "WarehouseToHome": 15.3,
         "HoursSpentOnApp": 3.9, "NumberOfDevicesRegistered": 3, "SatisfactionScore": 4, "NumberOfAddress": 2,
         "OrderCount": 35, "DaySinceLastOrder": 5, "CashbackAmount": 210.9},
        {"Age": 31, "Gender": "Male", "Tenure": 12.0, "PreferredLoginDevice": "Mobile", "CityTier": 1,
         "WarehouseToHome": 8.8,
         "HoursSpentOnApp": 3.0, "NumberOfDevicesRegistered": 2, "SatisfactionScore": 3, "NumberOfAddress": 1,
         "OrderCount": 18, "DaySinceLastOrder": 10, "CashbackAmount": 80.2},
    ]}

def test_health_check():
    respons = client.get("/health")
    data = respons.json()

    assert data["status_code"] == 200
    assert data["status"] == "Healthy"
    assert data["model_loaded"] == True

def test_prediction_valid_input(valid_sample_input):

    response = client.post("/predict", json=valid_sample_input)

    assert response.status_code == 200

    result_response = response.json()

    assert "prediction result" in result_response

    assert len(result_response["prediction result"]) == len(valid_sample_input["samples"])


def test_prediction_invalid_input():
    sample_payload = {"InvalidField": [
        {"Age": 29, "Gender": "Male", "Tenure": 5.0, "PreferredLoginDevice": "Desktop", "CityTier": 1,
         "WarehouseToHome": 7.2,
         "HoursSpentOnApp": 2.5, "NumberOfDevicesRegistered": 2, "SatisfactionScore": 3, "NumberOfAddress": 1,
         "OrderCount": 8, "DaySinceLastOrder": 15, "CashbackAmount": 45.2},
        {"Age": 45, "Gender": "Female", "Tenure": 36.0, "PreferredLoginDevice": "Mobile", "CityTier": 3,
         "WarehouseToHome": 20.1,
         "HoursSpentOnApp": 4.8, "NumberOfDevicesRegistered": 4, "SatisfactionScore": 5, "NumberOfAddress": 3,
         "OrderCount": 50, "DaySinceLastOrder": 2, "CashbackAmount": 320.5},
        {"Age": 22, "Gender": "Male", "Tenure": 2.0, "PreferredLoginDevice": "Mobile", "CityTier": 1,
         "WarehouseToHome": 5.0,
         "HoursSpentOnApp": 1.8, "NumberOfDevicesRegistered": 1, "SatisfactionScore": 2, "NumberOfAddress": 1,
         "OrderCount": 3, "DaySinceLastOrder": 25, "CashbackAmount": 10.1},
        {"Age": 39, "Gender": "Female", "Tenure": 24.0, "PreferredLoginDevice": "Desktop", "CityTier": 2,
         "WarehouseToHome": 15.3,
         "HoursSpentOnApp": 3.9, "NumberOfDevicesRegistered": 3, "SatisfactionScore": 4, "NumberOfAddress": 2,
         "OrderCount": 35, "DaySinceLastOrder": 5, "CashbackAmount": 210.9},
        {"Age": 31, "Gender": "Male", "Tenure": 12.0, "PreferredLoginDevice": "Mobile", "CityTier": 1,
         "WarehouseToHome": 8.8,
         "HoursSpentOnApp": 3.0, "NumberOfDevicesRegistered": 2, "SatisfactionScore": 3, "NumberOfAddress": 1,
         "OrderCount": 18, "DaySinceLastOrder": 10, "CashbackAmount": 80.2},
    ]}

    invalid_input = client.post("/predict", json=sample_payload)

    assert invalid_input.status_code == 422