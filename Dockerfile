# Paso 1: Usar una imagen base oficial de Python. 'slim' es una versión ligera.
FROM python:3.11-slim

# Paso 2: Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Paso 3: Copiar el archivo de requerimientos primero para aprovechar el cache de Docker
# Si no cambias los requerimientos, Docker no volverá a instalar las librerías en cada build.
COPY requirements.txt .

# Paso 4: Instalar todas las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Paso 5: Copiar todos los demás archivos de tu proyecto al contenedor
# (el código de la API, el script de creación, el CSV)
COPY . .

# Paso 6: Ejecutar el script para crear los modelos.
# Este es un paso CRUCIAL que se ejecuta al construir la imagen.
# Así, tu contenedor ya tendrá los archivos .pkl listos para ser usados.
RUN python crear_modelo_similitud.py

# Paso 7: Exponer el puerto 8000 para que sea accesible desde fuera del contenedor
EXPOSE 8000

# Paso 8: El comando para iniciar la aplicación cuando el contenedor se ejecute.
# El --host 0.0.0.0 es VITAL para que la API sea accesible desde tu máquina.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]