from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal
import numpy as np
import pandas as pd
import mlflow.pyfunc
from utils.config import MODEL_NAME, MLFLOW_TRACKING_URI

try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    MODEL_URI = f"models:/{MODEL_NAME}@champion"
    model = mlflow.pyfunc.load_model(MODEL_URI)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

app = FastAPI(
    title="Customer Churn Prediction API",
    description="A marketing agency has many customers that use their service to produce ads for the client/customer websites. They've noticed that they have quite a bit of churn in clients. They basically randomly assign account managers right now, but want you to create a machine learning model that will help predict which customers will churn (stop buying their service) so that they can correctly assign the customers most at risk to churn an account manager.",
    version="1.0",
)


class CustomerData(BaseModel):
    Age: int = Field(..., ge=18, le=100, description="Age of the customer")
    Total_Purchase: float = Field(..., ge=0, description="Total Ads Purchased")
    Account_Manager: Literal[0, 1] = Field(
        ..., description="Whether the customer has an account manager (0 or 1)"
    )
    Years: float = Field(..., ge=0, description="Number of years as a customer")
    Num_Sites: int = Field(..., ge=0, description="Number of websites the customer has")


@app.get("/")
def home():
    return {"message": "Churn Prediction API"}


@app.post("/predict")
def predict(data: CustomerData):
    try:
        # Convert input to numpy array
        input_df = pd.DataFrame(
            [
                {
                    "Age": data.Age,
                    "Total_Purchase": data.Total_Purchase,
                    "Account_Manager": data.Account_Manager,
                    "Years": data.Years,
                    "Num_Sites": data.Num_Sites,
                }
            ]
        )

        prediction = model.predict(input_df)

        result = "Churn" if prediction[0] == 1 else "Not Churn"

        return {"prediction": int(prediction[0]), "result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
