from typing import List, Literal
from pydantic import BaseModel, Field

class FeaturesInput(BaseModel):

    """
    اعتبار سنجی ویژگی های یک نمونه برای جلوگیری از ورود داده ها با ساختار نادرست به مدل
    """

    # محدود کردن بازه سنی بین 0 تا 89 سال
    Age: int = Field(gt=0, lt=90)
    # محدود کردن مقادیر gender به دو مقدار Male , Female
    Gender: object = Literal["Male", "Female"]
    Tenure: float
    PreferredLoginDevice: object = Literal["Desktop", "Mobile"]
    CityTier: int
    WarehouseToHome: float
    HoursSpentOnApp: float
    NumberOfDevicesRegistered: int
    SatisfactionScore: int
    NumberOfAddress: int
    OrderCount: int
    DaySinceLastOrder: int
    CashbackAmount: float

class BatchPredictionRequest(BaseModel):
    """
    تعیین کردن ساختار داده های ورودی به مدل و مقدار متناظر با فیلد sample لیستی از شی FeaturesInput است
    """
    samples: List[FeaturesInput]