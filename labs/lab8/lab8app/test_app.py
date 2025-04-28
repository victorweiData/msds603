# lab8app/test_app.py
import requests

url = "http://127.0.0.1:8000/predict"
payload = {
    "sepal_length": 5.1,
    "sepal_width":  3.5,
    "petal_length": 1.4,
    "petal_width":  0.2
}

resp = requests.post(url, json=payload)
print("Response:", resp.json())