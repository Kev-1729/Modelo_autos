from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib

app = FastAPI(
    title="API de Similitud de Autos",
    description="Provee recomendaciones de autos similares basadas en sus características.",
    version="1.0.0",
)

# Cargar los archivos pre-calculados al iniciar la aplicación
try:
    cosine_sim = joblib.load('cosine_sim_matrix.pkl')
    df_autos = pd.read_pickle('car_dataframe.pkl')
    print("Modelo de similitud y dataframe cargados.")
except FileNotFoundError:
    raise RuntimeError("Archivos de modelo no encontrados. Ejecuta 'crear_modelo_similitud.py' primero.")

@app.get("/")
def read_root():
    return {"message": "API de Similitud de Autos. Endpoints disponibles: /cars y /similar-cars/{car_id}"}

@app.get('/cars',
         tags=["Autos"],
         summary="Obtener la lista de todos los autos con su ID")
def get_all_cars():
    """
    Devuelve una lista de todos los autos con su `car_id`, `Marca`, `Modelo` y `Año`.
    Útil para que el frontend muestre una lista y pueda usar el `car_id` para pedir recomendaciones.
    """
    cols_to_return = ['car_id', 'Marca', 'Modelo', 'Año']
    return df_autos[cols_to_return].to_dict(orient='records')


@app.get('/similar-cars/{car_id}',
         tags=["Recomendaciones"],
         summary="Obtener autos similares a un auto específico por su ID")
def get_similar_cars(car_id: int, count: int = 5):
    """
    Recibe el `car_id` de un auto y devuelve una lista de los `count` autos más similares.
    """
    if car_id not in df_autos.index:
        raise HTTPException(status_code=404, detail=f"Auto con ID {car_id} no encontrado.")

    # 1. Obtener los puntajes de similitud del auto elegido contra todos los demás
    sim_scores = list(enumerate(cosine_sim[car_id]))

    # 2. Ordenar los autos basado en los puntajes de similitud (de mayor a menor)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 3. Obtener los IDs de los autos más similares, excluyendo el propio auto (que siempre será el primero)
    sim_scores = sim_scores[1:count+1]
    similar_car_indices = [i[0] for i in sim_scores]

    # 4. Devolver la información de los autos recomendados
    results = df_autos.iloc[similar_car_indices]
    
    return {
        "source_car": df_autos.iloc[car_id].to_dict(),
        "recommendations": results.to_dict(orient='records')
    }