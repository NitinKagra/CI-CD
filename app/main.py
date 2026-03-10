from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# load model
model = joblib.load("model.pkl")

class IrisInput(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "Iris prediction API"}

@app.post("/predict")
def predict(data: IrisInput):
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)

    return {"prediction": int(prediction[0])}