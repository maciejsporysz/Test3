# src/app.py

import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os

# Load the trained model
model = None
try:
    model_path = os.path.join("../model", "model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Model file not found at {model_path}. Please run train_model.py first.")

app = FastAPI(title="ML Prediction Service", version="1.0.0")


class InputData(BaseModel):
    data: list  # List of dictionaries representing feature values

@app.post("/predict")
#def predict():
def predict(input_data: InputData):
    #(input_data: InputDa
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Service unavailable.")

    try:
        # Convert input data to DataFrame
        print(f"Start")
        X_new = pd.DataFrame(input_data.data)
        # Get predictions
        prediction =model.predict(X_new)

        # Return predictions as a standard Python list
        return {"prediction": prediction.tolist()}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred during prediction.")



