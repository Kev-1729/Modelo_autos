from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Cargar modelo
modelo = joblib.load("modelo_autos.pkl")

# Crear app
app = FastAPI()

# Esquema de entradaz
class Auto(BaseModel):
    Marca: str
    Combustible: str
    Transmisión: str
    Kilometraje: float
    Antiguedad: float

@app.post("/predecir/")
def predecir_precio(auto: Auto):
    datos = pd.DataFrame([auto.dict()])
    prediccion = modelo.predict(datos)
    return {"precio_estimado": round(prediccion[0], 2)}
