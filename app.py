from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import joblib

app = FastAPI(
    title="API de Similitud de Autos",
    description="Provee recomendaciones de autos similares basadas en sus características.",
    version="2.0.0",
)

try:
    cosine_sim = joblib.load('cosine_sim_matrix.pkl')
    df_autos = pd.read_pickle('car_dataframe.pkl')
    print("Modelo de similitud y dataframe cargados correctamente.")
    df_autos['Marca_lower'] = df_autos['Marca'].str.lower()
    df_autos['Modelo_lower'] = df_autos['Modelo'].str.lower()
    print("Columnas de búsqueda en minúsculas creadas para búsquedas insensibles.")

except FileNotFoundError:
    raise RuntimeError("Archivos de modelo no encontrados. Ejecuta el build de Docker o el script 'crear_modelo_similitud.py' primero.")

@app.get("/")
def read_root():
    return {"message": "API de Similitud de Autos. Endpoints disponibles: /cars, /similar-cars/{car_id} y /similar-cars-by-model"}

@app.get('/similar-cars-by-model',
         tags=["Recomendaciones"],
         summary="Obtener autos similares usando Marca y Modelo")
def get_similar_cars_by_model(
    marca: str = Query(..., description="Marca del auto. Ejemplo: Toyota"),
    modelo: str = Query(..., description="Modelo del auto. Ejemplo: Yaris"),
    count: int = Query(5, description="Número de recomendaciones a devolver.")
):
    """
    Recibe una marca y un modelo, encuentra el auto más reciente que coincida
    y devuelve una lista de los autos más similares a él.
    """
    search_marca = marca.lower()
    search_modelo = modelo.lower()

    matching_cars = df_autos[
        (df_autos['Marca_lower'] == search_marca) &
        (df_autos['Modelo_lower'] == search_modelo)
    ]

    if matching_cars.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontraron autos para la marca '{marca}' y modelo '{modelo}'."
        )

    source_car = matching_cars.sort_values(by='Año', ascending=False).iloc[0]
    source_car_id = int(source_car['car_id'])

    sim_scores = list(enumerate(cosine_sim[source_car_id]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:count + 1]
    similar_car_indices = [i[0] for i in sim_scores]

    if not similar_car_indices:
        return {
            "source_car": source_car.to_dict(),
            "recommendations": [],
            "message": "Se encontró un auto de referencia, pero no hay otras recomendaciones similares."
        }

    results = df_autos.iloc[similar_car_indices]

    return {
        "source_car": source_car.to_dict(),
        "recommendations": results.to_dict(orient='records')
    }

