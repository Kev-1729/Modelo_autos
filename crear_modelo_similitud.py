import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

print("Iniciando la creación del modelo de similitud de contenido...")

# 1. Cargar datos de 'autos_neoauto.csv'
try:
    df_autos = pd.read_csv('autos_neoauto.csv')
except FileNotFoundError:
    print("Error: 'autos_neoauto.csv' no fue encontrado. Asegúrate de que esté en la misma carpeta.")
    exit()

# --- INICIO DE LA MEJORA: LIMPIEZA DE DATOS PROFUNDA ---

# Limpieza robusta y genérica para todas las columnas
# Esto previene los errores de JSON al reemplazar valores NaN.
for col in df_autos.columns:
    # Si la columna es de tipo 'object' (generalmente texto)
    if df_autos[col].dtype == 'object':
        df_autos[col].fillna('No especificado', inplace=True)
        df_autos[col] = df_autos[col].str.strip()
    # Si la columna es de tipo numérico (float, int)
    elif pd.api.types.is_numeric_dtype(df_autos[col]):
        # Llenamos los valores NaN (faltantes) con 0.
        # Puedes elegir otro valor si es más apropiado para tu caso (ej. -1).
        df_autos[col].fillna(0, inplace=True)

print("Limpieza inicial de NaN completada para todas las columnas.")

# --- FIN DE LA MEJORA ---


# Tratamiento específico y robusto para la columna 'Año'
df_autos['Año'] = pd.to_numeric(df_autos['Año'], errors='coerce')
df_autos.dropna(subset=['Año'], inplace=True) # Elimina filas si 'Año' no es un número válido
df_autos['Año'] = df_autos['Año'].astype(int)

# Reiniciar el índice para asegurar que sea continuo después de cualquier limpieza
df_autos.reset_index(drop=True, inplace=True)
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
cosine_sim_matrix = cosine_similarity(tfidf_matrix)
print("Matriz de similitud calculada exitosamente.")

# 4. Guardar la matriz y el dataframe procesado para uso de la API
joblib.dump(cosine_sim_matrix, 'cosine_sim_matrix.pkl')
df_autos.to_pickle('car_dataframe.pkl')

print("\n¡Proceso completado!")
print("Archivos 'cosine_sim_matrix.pkl' y 'car_dataframe.pkl' creados con datos limpios.")