from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI()

MODEL_PATH = 'logreg_model.joblib'  # Make sure this matches your saved pipeline path

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file '{MODEL_PATH}' not found.")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

class CustomerData(BaseModel):
    tenure: int
    InternetService: str
    OnlineSecurity: str
    TechSupport: str
    Contract: str
    PaymentMethod: str

@app.get("/")
def read_root():
    return {"message": "Telecom Churn Prediction API is running."}

@app.post("/predict")
def predict(data: CustomerData):
    try:
        input_data = {
            'tenure': [data.tenure],
            'InternetService': [data.InternetService],
            'OnlineSecurity': [data.OnlineSecurity],
            'TechSupport': [data.TechSupport],
            'Contract': [data.Contract],
            'PaymentMethod': [data.PaymentMethod]
        }
        input_df = pd.DataFrame(input_data)
        print("Input DataFrame columns:", input_df.columns)
        print("Input DataFrame:", input_df)
        prediction = model.predict(input_df)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        print("Prediction error:", e)
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)