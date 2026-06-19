from pydantic import BaseModel, Field
from typing import Optional


class ClientData(BaseModel):
    """
    Входные данные для предсказания конверсии.
    
    Примеры значений основаны на датасете UCI Bank Marketing.
    """
    age: int = Field(..., description="Возраст клиента", ge=18, le=100, example=42)
    job: str = Field(..., description="Профессия", example="management")
    marital: str = Field(..., description="Семейное положение", example="married")
    education: str = Field(..., description="Образование", example="tertiary")
    default: str = Field(..., description="Есть ли дефолт по кредиту", example="no")
    balance: int = Field(..., description="Средний баланс счёта", ge=0, example=1500)
    housing: str = Field(..., description="Есть ли ипотека", example="yes")
    loan: str = Field(..., description="Есть ли потребительский кредит", example="no")
    contact: str = Field(..., description="Тип коммуникации", example="cellular")
    day: int = Field(..., description="День месяца", ge=1, le=31, example=15)
    month: str = Field(..., description="Месяц", example="may")
    duration: int = Field(..., description="Длительность звонка (секунды)", ge=0, example=250)
    campaign: int = Field(..., description="Количество контактов в этой кампании", ge=1, example=2)
    pdays: int = Field(..., description="Дней прошло с прошлого контакта (-1 если не было)", ge=-1, example=95)
    previous: int = Field(..., description="Количество контактов до этой кампании", ge=0, example=1)
    poutcome: str = Field(..., description="Результат прошлой кампании", example="success")


class PredictionResponse(BaseModel):
    """
    Ответ сервиса с предсказанием.
    """
    prediction: int = Field(..., description="Предсказанный класс (0 или 1)")
    probability: float = Field(..., description="Вероятность класса", ge=0.0, le=1.0)
    model_version: str = Field(..., description="Версия модели")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.87,
                "model_version": "v1"
            }
        }


class HealthResponse(BaseModel):
    """
    Ответ эндпоинта health.
    """
    status: str
    model_version: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "model_version": "v1"
            }
        }