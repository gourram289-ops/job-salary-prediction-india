import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field


# =====================================================
# Pydantic Schemas for Input Validation
# =====================================================

class JobApplicationData(BaseModel):
    job_title: str = Field(..., example="Software Engineer")
    education: str = Field(..., example="B.Tech")
    primary_skill: str = Field(..., example="Python")
    city: str = Field(..., example="Bengaluru")
    state: str = Field(..., example="Karnataka")
    company: str = Field(..., example="Infosys")
    company_size: str = Field(..., example="10000+")
    work_mode: str = Field(..., example="Hybrid")
    experience_years: float = Field(..., ge=0, example=20)
    skills_count: int = Field(..., ge=1, example=20)

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_title": "Data Scientist",
                "education": "M.Tech",
                "primary_skill": "Machine Learning",
                "city": "Hyderabad",
                "state": "Telangana",
                "company": "Microsoft",
                "company_size": "10000+",
                "work_mode": "Remote",
                "experience_years": 3.0,
                "skills_count": 8
            }
        }
    }


class PredictionResponse(BaseModel):
    predicted_salary_lpa: float
    currency: str = "INR (Lakhs Per Annum)"


# =====================================================
# App Initialization & Model Loading
# =====================================================

app = FastAPI(
    title="India Tech Salary Predictor API",
    description="API for predicting tech salaries in India using a trained Scikit-Learn/XGBoost pipeline.",
    version="1.0.0"
)

MODEL_PATH = "salary_model.pkl"
model_pipeline = None


@app.on_event("startup")
def load_model():
    """Load the machine learning pipeline on server startup."""
    global model_pipeline
    try:
        model_pipeline = joblib.load(MODEL_PATH)
        print(f"✅ Model successfully loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise RuntimeError(f"Could not load the model file at {MODEL_PATH}")


# =====================================================
# API Endpoints
# =====================================================

@app.get("/", tags=["Health"])
def health_check():
    """Sanity check endpoint to verify API and model status."""
    return {
        "status": "online",
        "model_loaded": model_pipeline is not None
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_salary(data: JobApplicationData):
    """
    Predict salary in LPA based on job, company, and candidate profile.
    """
    if not model_pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or unavailable."
        )

    try:
        # 1. Convert Pydantic object to a dictionary
        input_dict = data.model_dump()
        
        # 2. Convert dictionary to DataFrame (ColumnTransformer expects a DataFrame with specific column names)
        input_df = pd.DataFrame([input_dict])
        
        # 3. Generate prediction using the entire preprocessing + model pipeline
        prediction = model_pipeline.predict(input_df)
        
        # 4. Extract scalar value and round to 2 decimal places
        salary_lpa = round(float(prediction[0]), 2)
        
        return PredictionResponse(predicted_salary_lpa=salary_lpa)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction error: {str(e)}"
        )