from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import joblib
import pandas as pd
import pyagrum as gum
import feature_extraction as fe

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add this middleware to disable CORS restrictions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)
# Load trained model and resources
model_data = joblib.load("models/models.joblib")
model_ml = model_data["model"]
scaler = model_data["scaler"]

# Load Bayesian Network and initialize inference engine
bn = gum.loadBN("models/bn_model.bif")
ie = gum.LazyPropagation(bn)

# Request models
class Measurements(BaseModel):
    rr_filter: List[float] = Field(..., min_items=1)
    gsr_filter: List[float] = Field(..., min_items=1)
    temperature_filter: List[float] = Field(..., min_items=1)
    hr_filter: List[float] = Field(..., min_items=1)

class UserProfile(BaseModel):
    gentleness: int
    fairness: int
    greed_avoidance: int
    modesty: int
    sincerity: int
    liveliness: int
    sociability: int
    social_boldness: int
    openess: int
    dependence: int
    diligence: int
    prudence: int
    flexibility: int
    patience: int
    Age: int
    Sex: int

class RequestBody(BaseModel):
    measurements: Measurements
    user_profile: UserProfile

@app.post("/analyze")
async def analyze_data(body: RequestBody):
    # Build input DataFrame
    m = body.measurements
    n = len(m.rr_filter)
    df_slice = pd.DataFrame({
        "rr_filter": m.rr_filter,
        "gsr_filter": m.gsr_filter,
        "temperature_filter": m.temperature_filter,
        "hr_filter": m.hr_filter,
    })

    # Feature extraction and ML prediction
    features = fe.extract_all_features(df_slice)
    scaled , _ = fe.combine_and_standardize(features, scaler=scaler, fit_scaler=False)
    X_test = scaled.drop('rr_sd1', axis=1)
    y_pred = model_ml.predict(X_test)
    workload_marker = int(y_pred[0])

    # Prepare Bayesian evidence
    evs = body.user_profile.model_dump()
    evs["Physiological_workload_markers"] = workload_marker

    # Inference
    ie.setEvidence(evs)
    ie.makeInference()
    posterior = ie.posterior("Workload")
    load_prob = float(posterior[1])

    return {
        "load_assessment": int(load_prob > 0.5)
    }
