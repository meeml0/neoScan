from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import os
from model import ModelInference
from utils import mask_to_base64, validate_image_format

app = FastAPI(title="Kanser Hücresi Tespit API", version="1.0.0")

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (frontend)
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Model instance
model = ModelInference()

@app.get("/")
async def root():
    """Ana sayfa - frontend'i serve et"""
    frontend_file = os.path.join(frontend_path, "index.html")
    if os.path.exists(frontend_file):
        return FileResponse(frontend_file)
    return {"message": "Kanser Hücresi Tespit API", "docs": "/docs"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API çalışıyor"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Sadece segmentasyon maskı döndürür"""
    if not validate_image_format(file.filename):
        raise HTTPException(status_code=400, detail="Desteklenmeyen dosya formatı")
    
    try:
        image_bytes = await file.read()
        results = model.predict(image_bytes)
        mask_b64 = mask_to_base64(results["segmentation_mask"])
        
        return {
            "success": True,
            "filename": file.filename,
            "results": {
                "detected_cells": results["detected_cells"],
                "model_accuracy": results["model_accuracy"],
                "segmentation_mask": mask_b64
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin hatası: {str(e)}")

@app.post("/predict_with_overlay")
async def predict_with_overlay(file: UploadFile = File(...)):
    """Maske ve görselleştirme ile birlikte sonuç döndürür"""
    if not validate_image_format(file.filename):
        raise HTTPException(status_code=400, detail="Desteklenmeyen dosya formatı")
    try:
        image_bytes = await file.read()
        results = model.predict(image_bytes)
        # Görselleştirme base64 ve oranı döndür
        return {
            "success": True,
            "filename": file.filename,
            "results": {
                "predicted_ratio": results["predicted_ratio"],
                "model_accuracy": results["model_accuracy"],
                "visualization": results["visualization"],
                "detected_cells": results["detected_cells"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin hatası: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)