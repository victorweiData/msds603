# lab8app/lab8app.py
import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel
import os, mlflow
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5002"

# load the latest version of our registered model
mlflow.set_tracking_uri("http://127.0.0.1:5002")
model = mlflow.pyfunc.load_model("models:/iris-classifier/latest")

app = FastAPI()

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width:  float
    petal_length: float
    petal_width:  float

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: IrisFeatures):
    X = [[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]]
    pred = model.predict(X)[0]
    return {"prediction": int(pred)}