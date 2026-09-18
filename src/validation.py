from typing import List, Literal
from pydantic import BaseModel, Field

class FeaturesInput(BaseModel):
    Age: int = Field(gt=0, lt=90)
    Gender: object = Literal["Male", "Female"]
    Tenure: float = Field()
    PreferredLoginDevice: object = Literal["Desktop", "Mobile"]
    CityTier: int = Field()
    WarehouseToHome: float = Field()
    HoursSpentOnApp: float = Field()
    NumberOfDevicesRegistered: int = Field()
    SatisfactionScore: int = Field()
    NumberOfAddress: int = Field()
    OrderCount: int = Field()
    DaySinceLastOrder: int = Field()
    CashbackAmount: float = Field()

class BatchPredictionRequest(BaseModel):
    samples: List[FeaturesInput]