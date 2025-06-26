import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib

print("Iniciando la creación del modelo de similitud de contenido...")

# 1. Cargar y limpiar datos de 'autos_neoauto.csv'
try:
    df_autos = pd.read_csv('autos_neoauto.csv')
except FileNotFoundError:
    print("Error: 'autos_neoauto.csv' no fue encontrado. Asegúrate de que esté en la misma carpeta.")
    exit()

# Limpieza robusta de datos
for col in ['Marca', 'Modelo', 'Transmisión', 'Combustible']:
    if col in df_autos.columns:
        df_autos[col] = df_autos[col].fillna('No especificado').str.strip()
df_autos['Año'] = pd.to_numeric(df_autos['Año'], errors='coerce')
df_autos.dropna(subset=['Año'], inplace=True)
df_autos['Año'] = df_autos['Año'].astype(int)
df_autos.reset_index(drop=True, inplace=True)
# El índice del dataframe será nuestro 'car_id'
df_autos['car_id'] = df_autos.index 

print(f"DataFrame cargado y limpiado. {len(df_autos)} autos procesados.")

# 2. Crear el "perfil" de cada auto combinando características
df_autos['caracteristicas_texto'] = (df_autos['Marca'] + ' ' + df_autos['Modelo'] + ' ' +
                                    df_autos['Transmisión'] + ' ' + df_autos['Combustible'] + ' ' +
                                    df_autos['Año'].astype(str))

# Vectorizar el texto
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df_autos['caracteristicas_texto'])

# 3. Calcular la matriz de similitud del coseno
# Esto crea una gran tabla donde cada celda [i, j] dice qué tan similar es el auto 'i' al auto 'j'
cosine_sim_matrix = cosine_similarity(tfidf_matrix)
print("Matriz de similitud calculada exitosamente.")

# 4. Guardar la matriz y el dataframe procesado para uso de la API
joblib.dump(cosine_sim_matrix, 'cosine_sim_matrix.pkl')
df_autos.to_pickle('car_dataframe.pkl')

print("\n¡Proceso completado!")
print("Archivos 'cosine_sim_matrix.pkl' y 'car_dataframe.pkl' creados.")