from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import sys
from app.service.predict import process_image, load_model

app = FastAPI(title="PCMMD Backend", 
              description="Backend for Cell Morphology and Migration Dynamics Analysis")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "https://pcmmd-frontend.vercel.app"],  # Thêm domain Vercel của bạn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
model = None

@app.on_event("startup")
async def startup_event():
    global model
    try:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "yolo10l_final.pt")
        model = load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

@app.get("/")
def read_root():
    return {"status": "online", "message": "PCMMD API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Check file extension
    allowed_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not supported. Please upload {', '.join(allowed_extensions)}"
        )
    
    try:
        contents = await file.read()
        results = process_image(model, contents)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/metrics")
def get_metrics():
    """Return available metrics for cell migration analysis"""
    metrics = {
        "morphological": [
            {"id": "area", "name": "Cell Area", "unit": "μm²", "description": "Total area occupied by the cell"},
            {"id": "perimeter", "name": "Cell Perimeter", "unit": "μm", "description": "Perimeter of the cell boundary"},
            {"id": "aspect_ratio", "name": "Aspect Ratio", "unit": "", "description": "Ratio of major to minor axis length"},
            {"id": "circularity", "name": "Circularity", "unit": "", "description": "How closely the cell resembles a perfect circle"},
            {"id": "solidity", "name": "Solidity", "unit": "", "description": "Measure of cell convexity"},
        ],
        "migratory": [
            {"id": "velocity", "name": "Migration Velocity", "unit": "μm/min", "description": "Speed of cell movement"},
            {"id": "displacement", "name": "Net Displacement", "unit": "μm", "description": "Straight-line distance traveled"},
            {"id": "directionality", "name": "Directionality Ratio", "unit": "", "description": "Measure of migration persistence"},
            {"id": "msd", "name": "Mean Square Displacement", "unit": "μm²", "description": "Quantifies exploration area over time"},
        ]
    }
    return metrics

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
