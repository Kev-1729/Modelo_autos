from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import joblib

app = FastAPI(
    title="API de Similitud de Autos",
    description="Provee recomendaciones de autos similares basadas en sus características.",
    version="2.0.0", # ¡Subimos de versión por el gran cambio!
)

# Cargar los archivos pre-calculados al iniciar la aplicación
try:
    cosine_sim = joblib.load('cosine_sim_matrix.pkl')
    df_autos = pd.read_pickle('car_dataframe.pkl')
    print("Modelo de similitud y dataframe cargados correctamente.")
    # Asegurar que las columnas de texto sean consistentes para la búsqueda
    df_autos['Marca_lower'] = df_autos['Marca'].str.lower()
    df_autos['Modelo_lower'] = df_autos['Modelo'].str.lower()
    print("Columnas de búsqueda en minúsculas creadas para búsquedas insensibles.")

except FileNotFoundError:
    raise RuntimeError("Archivos de modelo no encontrados. Ejecuta el build de Docker o el script 'crear_modelo_similitud.py' primero.")

@app.get("/")
def read_root():
    return {"message": "API de Similitud de Autos. Endpoints disponibles: /cars, /similar-cars/{car_id} y /similar-cars-by-model"}

# --- INICIO DEL NUEVO ENDPOINT ---

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
    # 1. Normalizar las entradas para hacer la búsqueda insensible a mayúsculas/minúsculas
    search_marca = marca.lower()
    search_modelo = modelo.lower()

    # 2. Buscar todos los autos que coincidan con la marca y modelo
    matching_cars = df_autos[
        (df_autos['Marca_lower'] == search_marca) &
        (df_autos['Modelo_lower'] == search_modelo)
    ]

    # 3. Si no se encuentra ningún auto, devolver un error 404
    if matching_cars.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No se encontraron autos para la marca '{marca}' y modelo '{modelo}'."
        )

    # 4. Seleccionar el auto más reciente como nuestro auto de referencia
    source_car = matching_cars.sort_values(by='Año', ascending=False).iloc[0]
    source_car_id = int(source_car['car_id'])

    # 5. Reutilizar la lógica de similitud existente con el ID del auto de referencia
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

