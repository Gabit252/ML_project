import pickle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

app = FastAPI(title="Predict cancer type API")

with open("Cancer_Data_model.pkl", "rb") as f:
    model = pickle.load(f)
    
class PredictionSettings(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float
  
@app.get("/")
def read_root():
    return {"message": "Welcome to Cancer Type Prediciton API"}

@app.post("/predict")
def predict(pre_set: PredictionSettings):
    data = pre_set.dict()
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return {"Diagnosis": float(prediction)}

if __name__=="__main__":
    uvicorn.run(app, host='0.0.0.0', port=8001)
    
