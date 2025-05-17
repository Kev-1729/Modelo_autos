import requests

url = "http://127.0.0.1:8000/predecir/"

data = {
    "Marca": "Toyota",
    "Combustible": "Gasolina",
    "Transmisión": "Automática",
    "Kilometraje": 50000,
    "Antiguedad": 3
}

response = requests.post(url, json=data)
print(response.json())