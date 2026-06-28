# main.py
# Autor: CobraKai
# Fecha: 2026-06-18
# Descripción: Microservicio FastAPI para predecir la ocupación hotelera.
# Expone un endpoint /predict/occupancy para obtener predicciones y /health para monitoreo.

import os
import joblib
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- 1. Definición de Modelos de Datos (Pydantic) ---

class PredictionRequest(BaseModel):
    """Esquema para la solicitud de predicción por rango de fechas."""
    fecha_inicio: str = Field(..., example="2026-07-01")
    fecha_fin: str = Field(..., example="2026-09-30")
    average_base_price: float = Field(120.0, example=120.0)
    season_percentage: float = Field(15.0, example=15.0)
    offer_percentage: float = Field(0.0, example=0.0)

class DailyPrediction(BaseModel):
    """Esquema para la predicción de un solo día."""
    date: str
    predicted_occupancy: int
    lower_bound: int
    upper_bound: int
    predicted_revenue: float

class PredictionResponse(BaseModel):
    """Esquema para la respuesta de predicción."""
    predictions: List[DailyPrediction]
    
class HealthResponse(BaseModel):
    """Esquema para la respuesta del health check."""
    status: str
    model_loaded: bool


# --- 2. Inicialización de la Aplicación y Carga del Modelo ---

# Rutas a los artefactos del modelo
MODEL_PATH = os.path.join("models", "occupancy_model.joblib")
FEATURES_PATH = os.path.join("models", "model_features.joblib")

# Variables globales para almacenar el modelo y las características
model = None
model_features = None

app = FastAPI(
    title="MAREA - Hotel Occupancy Prediction API",
    description="Un microservicio para predecir la ocupación de habitaciones de hotel.",
    version="1.0.0"
)

@app.on_event("startup")
def load_model():
    """
    Carga el modelo y las características en memoria al iniciar la aplicación.
    Si los archivos no existen, el modelo permanecerá como None y los endpoints fallarán.
    """
    global model, model_features
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
            model = joblib.load(MODEL_PATH)
            model_features = joblib.load(FEATURES_PATH)
            print("--- Modelo y características cargados exitosamente. ---")
        else:
            print("--- ADVERTENCIA: No se encontraron los archivos del modelo. El endpoint de predicción no funcionará. ---")
            print("--- Ejecute `python train.py` para generar los artefactos del modelo. ---")
    except Exception as e:
        print(f"--- ERROR CRÍTICO: No se pudo cargar el modelo. Causa: {e} ---")
        model = None
        model_features = None

# --- 3. Configuración de CORS ---

# Configuración abierta para un entorno de desarrollo.
# Para producción, se recomienda restringir los orígenes.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas las origenes
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos
    allow_headers=["*"],  # Permite todas las cabeceras
)

# --- 4. Definición de Endpoints ---

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """
    Endpoint de salud. Verifica si el modelo está cargado y operativo.
    Retorna 503 Service Unavailable si el modelo no está listo.
    """
    is_model_ready = model is not None and model_features is not None
    if not is_model_ready:
        raise HTTPException(
            status_code=503, 
            detail="Servicio no disponible: El modelo de predicción no está cargado."
        )
    return {"status": "ok", "model_loaded": True}

@app.post("/predict/occupancy", response_model=PredictionResponse, tags=["Predictions"])
def predict_occupancy(request: PredictionRequest):
    """
    Realiza una predicción de ocupación para un rango de fechas dado.
    """
    if model is None or model_features is None:
        raise HTTPException(
            status_code=503, 
            detail="El modelo no está cargado. Por favor, entrene el modelo primero."
        )

    try:
        start_date = datetime.strptime(request.fecha_inicio, "%Y-%m-%d")
        end_date = datetime.strptime(request.fecha_fin, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400, 
            detail="Formato de fecha inválido. Use YYYY-MM-DD para fecha_inicio y fecha_fin."
        )

    if start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="fecha_inicio no puede ser mayor que fecha_fin."
        )

    results: List[DailyPrediction] = []

    delta = end_date - start_date

    for i in range(delta.days + 1):
        date_obj = start_date + timedelta(days=i)
        current_date_str = date_obj.strftime("%Y-%m-%d")
            
        data = {
            'average_base_price': [request.average_base_price],
            'season_percentage': [request.season_percentage],
            'offer_percentage': [request.offer_percentage],
            'is_holiday': [1 if date_obj.weekday() >= 5 else 0],
            'month': [date_obj.month],
            'day_of_week': [date_obj.weekday()],
            'is_weekend': [1 if date_obj.weekday() >= 5 else 0]
        }
        inference_df = pd.DataFrame(data)
        
        # Asegurarse de que las columnas estén en el orden correcto
        inference_df = inference_df[model_features]

        # Realizar la predicción
        prediction = model.predict(inference_df)[0]
        
        predicted_val = max(0, int(round(prediction))) # Asegurar no negativos
        
        # Calcular ingreso estimado
        # Formula: Occupied_Rooms * Base_Price * (1 + Season/100) * (1 - Offer/100)
        revenue = predicted_val * request.average_base_price * (1 + request.season_percentage / 100) * (1 - request.offer_percentage / 100)
        
        lower_bound = int(round(predicted_val * 0.85))
        upper_bound = int(round(predicted_val * 1.15))

        results.append(
            DailyPrediction(
                date=current_date_str,
                predicted_occupancy=predicted_val,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                predicted_revenue=round(revenue, 2)
            )
        )

    return PredictionResponse(predictions=results)

# --- 5. Ejecución del Servidor (para desarrollo) ---
# En producción, se usaría un gestor de procesos como Gunicorn con Uvicorn.
# Ejemplo: uvicorn main:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    print("Iniciando servidor FastAPI en http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
