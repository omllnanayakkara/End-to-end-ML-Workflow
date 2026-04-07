import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
from src.data_validation import validate_transaction_data

mlflow.set_tracking_uri("http://localhost:5000")

try:
    model = mlflow.sklearn.load_model("models:/fraud-detection-model@champion")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)
print("Encoder loaded successfully!")


app = FastAPI(
    title="Fraud Detection API (Validated)",
    description="""
    Fraud detection API with input validation.
    
    All inputs are validated before prediction:
    - amount: Must be positive and below $50,000
    - hour: Must be 0-23
    - day_of_week: Must be 0-6
    - merchant_category: Must be one of: grocery, restaurant, retail, online, travel
    
    Invalid inputs return HTTP 400 with detailed error messages.
    """,
    version="3.0.0"
)

class Transaction(BaseModel):
    amount: float = Field(
        ...,
        description="Transaction amount in dollars",
        example=123.45
    )
    hour: int = Field(
        ...,
        description="Hour of the day (0-23)",
        ge=0,
        le=23,
        example=14
    )
    day_of_week: int = Field(
        ...,
        description="Day of the week (0=Monday, 6=Sunday)",
        ge=0,
        le=6,
        example=2
    )
    merchant_category: str = Field(
        ...,
        description="Merchant category (grocery, restaurant, retail, online, travel)",
        example="online"
    )


class PredictionResponse(BaseModel):
    is_fraud: bool = Field(
        ...,
        description="Whether the transaction is predicted to be fraudulent"
    )
    probability: float = Field(
        ...,
        description="Probability of the transaction being fraudulent",
        ge=0.0,
        le=1.0
    )
    validation_passed: bool = True

class ValidationErrorResponse(BaseModel):
    detail: dict

@app.post("/predict", response_model=PredictionResponse, responses={400: {"model": ValidationErrorResponse}})
def predict(transaction: Transaction):
    data = transaction.model_dump()

    validation = validate_transaction_data(data)
    print(f"validation result: {validation}")
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail={
            "message": "Invalid input data", 
            "errors": validation["errors"],
            "input": data
        })

    try:
        data["merchant_encoded"] = encoder.transform([data["merchant_category"]])[0]
    except ValueError:
        data["merchant_encoded"] = 0

    import pandas as pd 
    
    X = pd.DataFrame([{
        "amount": data["amount"],
        "hour": data["hour"],
        "day_of_week": data["day_of_week"],
        "merchant_category": data["merchant_encoded"]
    }])

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return PredictionResponse(is_fraud=prediction, probability=round(float(probability), 4))

@app.get("/health")
def health_check():
    return {
        "status": "ok", "message": "API is healthy and ready to serve predictions.",
        "model_loaded": model is not None
    }

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Naive Fraud Detection API! Use the /predict endpoint to check transactions.",
        "version": "1.0.0",
        "docs": "/docs",
        "example_transaction": {
            "amount": 123.45,
            "hour": 14,
            "day_of_week": 2,
            "merchant_category": "online"
        }
    }