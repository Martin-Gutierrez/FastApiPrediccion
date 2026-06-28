# train.py
# Autor: Gemini-CLI
# Fecha: 2026-06-18
# Descripción: Script para entrenar un modelo de predicción de ocupación hotelera.
# Este script carga datos, realiza ingeniería de características, entrena un modelo
# XGBoost y lo exporta junto con las características utilizadas.

import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path
from typing import List, Tuple

# --- 1. Configuración de Rutas y Constantes ---
# Se utilizan objetos Path para compatibilidad entre sistemas operativos.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
EXTERNAL_DATA_PATH = DATA_DIR / "external_hotel_data.csv"
MODEL_PATH = MODEL_DIR / "occupancy_model.joblib"
FEATURES_PATH = MODEL_DIR / "model_features.joblib"

# --- 2. Generación de Datos Sintéticos ---
def generate_synthetic_data(path: Path) -> pd.DataFrame:
    """
    Genera un dataset sintético alineado a la base de datos (con precios base, ofertas y temporadas).
    """
    print("Generando datos sintéticos...")
    date_range = pd.date_range(start="2024-01-01", end="2026-12-31", freq="D")
    num_days = len(date_range)

    data = {
        "date": date_range,
        "occupied_rooms": (
            60 
            + np.sin(np.arange(num_days) * (2 * np.pi / 365.25)) * 40 
            + np.random.randint(-15, 15, num_days)
            + (date_range.month.isin([6, 7, 8, 12])) * 20
        ).astype(int),
        "average_base_price": (
            100 
            + np.linspace(0, 10, num_days) 
            + np.random.uniform(-5, 5, num_days)
        ).round(2),
        "season_percentage": np.random.choice(
            [0, 15, 30], 
            num_days, 
            p=[0.7, 0.2, 0.1]
        ),
        "offer_percentage": np.random.choice(
            [0, 10, 20], 
            num_days, 
            p=[0.8, 0.15, 0.05]
        ),
        "is_holiday": np.random.choice(
            [0, 1], 
            num_days, 
            p=[0.9, 0.1]
        )
    }
    df = pd.DataFrame(data)
    df.loc[df["date"].dt.dayofweek >= 5, "is_holiday"] = np.random.choice(
        [0, 1],
        size=(df["date"].dt.dayofweek >= 5).sum(),
        p=[0.7, 0.3]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Datos sintéticos guardados en: {path}")
    return df

# --- 3. Ingeniería de Características ---
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade características de series temporales basadas en la columna 'date'.

    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        pd.DataFrame: DataFrame con las nuevas características.
    """
    print("Realizando ingeniería de características...")
    # Asegurarse de que la columna 'date' es de tipo datetime
    df['date'] = pd.to_datetime(df['date'])
    
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek  # Lunes=0, Domingo=6
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int) # Sábado o Domingo

    return df

# --- 4. Flujo Principal de Entrenamiento ---
def main():
    """
    Orquesta el proceso de carga de datos, entrenamiento y guardado del modelo.
    """
    print("--- Iniciando el Proceso de Entrenamiento del Modelo ---")

    # Crear directorios necesarios si no existen
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    # Cargar o generar datos
    try:
        hotel_data = pd.read_csv(EXTERNAL_DATA_PATH)
        print(f"Datos cargados exitosamente desde: {EXTERNAL_DATA_PATH}")
    except FileNotFoundError:
        print(f"No se encontró el archivo en '{EXTERNAL_DATA_PATH}'.")
        hotel_data = generate_synthetic_data(EXTERNAL_DATA_PATH)

    # Aplicar ingeniería de características
    processed_data = feature_engineering(hotel_data.copy())

    # Definir variables predictoras (X) y objetivo (y)
    FEATURES = [
        'average_base_price',
        'season_percentage',
        'offer_percentage',
        'is_holiday',
        'month',
        'day_of_week',
        'is_weekend'
    ]
    TARGET = 'occupied_rooms'
    
    X = processed_data[FEATURES]
    y = processed_data[TARGET]
    
    print(f"Características utilizadas para el entrenamiento: {FEATURES}")

    # Entrenar el modelo XGBoost Regressor
    print("Entrenando el modelo XGBoost...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1 # Usar todos los cores disponibles
    )
    model.fit(X, y)
    print("Entrenamiento completado.")

    # Guardar el modelo y la lista de características
    print(f"Guardando modelo en: {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)

    print(f"Guardando lista de características en: {FEATURES_PATH}")
    joblib.dump(FEATURES, FEATURES_PATH)
    
    print("--- Proceso de Entrenamiento Finalizado Exitosamente ---")

if __name__ == "__main__":
    main()
