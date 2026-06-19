from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
import joblib
import time
import uuid
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.service.schemas import ClientData, PredictionResponse, HealthResponse
from src.service.logger import logger

app = FastAPI(
    title="Bank Conversion Prediction Service",
    description="Сервис прогнозирования вероятности оформления срочного депозита",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
model_version = "v1"

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Логирует все HTTP-запросы с временем выполнения."""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    try:
        response = await call_next(request)
        latency_ms = (time.time() - start_time) * 1000
        if request.url.path in ["/predict", "/health"]:
            logger.info(
                f"{request.method} {request.url.path} "
                f"status={response.status_code} "
                f"latency={latency_ms:.1f}ms "
                f"request_id={request_id}"
            )
        
        return response
        
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(
            f"{request.method} {request.url.path} "
            f"error={str(e)} "
            f"latency={latency_ms:.1f}ms "
            f"request_id={request_id}"
        )
        raise

@app.on_event("startup")
async def load_model():
    """Загружает модель при старте сервиса."""
    global model
    
    model_path = config.models_dir / 'pipeline_v1.pkl'
    
    if not model_path.exists():
        logger.error(f"Модель не найдена: {model_path}")
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    
    model = joblib.load(model_path)
    logger.info(f"Модель загружена: {model_path}")
    logger.info(f"Тип модели: {type(model).__name__}")

@app.get("/")
async def root():
    """Корневой эндпоинт с информацией о сервисе."""
    return {
        "service": "Bank Conversion Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Проверка работоспособности сервиса.
    Возвращает статус 'ok' если модель загружена.
    """
    if model is None:
        logger.warning("/health: модель не загружена")
        return HealthResponse(status="error", model_version=None)
    
    return HealthResponse(status="ok", model_version=model_version)


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: ClientData, request: Request):
    """
    Предсказание вероятности конверсии клиента.
    
    Принимает данные клиента и возвращает предсказание (0 или 1) 
    и вероятность принадлежности к классу 1.
    """
    start_time = time.time()
    
    try:
        import pandas as pd
        df = pd.DataFrame([data.dict()])
        
        # Делаем предсказание
        prediction = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0][1])
        latency_ms = (time.time() - start_time) * 1000
        logger.info(
            f"/predict prediction={prediction} "
            f"probability={probability:.4f} "
            f"latency={latency_ms:.1f}ms"
        )
        
        return PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            model_version=model_version
        )
        
    except ValidationError as e:
        logger.error(f"/predict validation failed: {e}")
        raise HTTPException(status_code=400, detail=f"Validation error: {e}")
    
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        logger.error(f"/predict error={str(e)} latency={latency_ms:.1f}ms")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host=config.get('service.host', '0.0.0.0'),
        port=config.get('service.port', 8000),
        reload=False
    )