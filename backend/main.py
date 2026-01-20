from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import pickle
import numpy as np
import pandas as pd
import os
from typing import List, Dict
import json

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and route data on startup
@app.on_event("startup")
def load_assets():
    try:
        # Load model
        model_path = os.path.join(os.path.dirname(__file__), 'safety_model.pkl')
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                app.state.model = model_data.get('model')
                app.state.scaler = model_data.get('scaler')
        except FileNotFoundError:
            print("Model file not found - proceeding without model")
        
        # Load route data
        routes_path = os.path.join(os.path.dirname(__file__), 'borivali_dahisar_safety_routes_updated.csv')
        try:
            app.state.routes_df = pd.read_csv(routes_path)
        except FileNotFoundError:
            print("Routes file not found - proceeding without routes")
        
        print("Assets loaded successfully")
    except Exception as e:
        print(f"Error loading assets: {e}")

@app.get("/")
async def root():
    return {"message": "FastAPI backend is running"}

@app.post("/get_routes/")
async def get_routes(source: str = Form(...), destination: str = Form(...)):
    try:
        # Example routes with coordinates
        routes = [
            {
                "path": "Eksar -> Mandapeshwar",
                "distance": 0.8,
                "safety_score": 0.65,
                "safety_level": "Moderate",
                "crime_reports": 3,
                "police_distance": 0.5,
                "cctv_count": 2,
                "is_best_route": True,
                "safety_tip": "This route is moderately safe. Stay alert during night hours.",
                "coordinates": [
                    {"location": "Eksar", "latitude": 19.2359, "longitude": 72.8521},
                    {"location": "Mandapeshwar", "latitude": 19.2382, "longitude": 72.8545}
                ]
            },
            {
                "path": "Eksar -> IC Colony -> Mandapeshwar",
                "distance": 1.2,
                "safety_score": 0.45,
                "safety_level": "Risky",
                "crime_reports": 8,
                "police_distance": 0.9,
                "cctv_count": 1,
                "is_best_route": False,
                "safety_tip": "This route has higher risk. Consider traveling with others.",
                "coordinates": [
                    {"location": "Eksar", "latitude": 19.2359, "longitude": 72.8521},
                    {"location": "IC Colony", "latitude": 19.2415, "longitude": 72.8598},
                    {"location": "Mandapeshwar", "latitude": 19.2382, "longitude": 72.8545}
                ]
            }
        ]

        # Filter and sort routes
        filtered_routes = []
        for route in routes:
            path_parts = route["path"].split(" -> ")
            if path_parts[0].strip().lower() == source.strip().lower() and path_parts[-1].strip().lower() == destination.strip().lower():
                filtered_routes.append(route)

        filtered_routes.sort(key=lambda x: x["safety_score"], reverse=True)
        if filtered_routes:
            for route in filtered_routes:
                route["is_best_route"] = (route == filtered_routes[0])
        
        return JSONResponse(content={"routes": filtered_routes})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

