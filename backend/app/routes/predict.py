from fastapi import APIRouter, File, UploadFile, HTTPException
from app.schemas import PredictionResponse, HealthResponse
from app.services.model_service import model_service
from app.services.advice_engine import get_advice
from app.utils.image import preprocess_image
from app.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "ok",
        "model_loaded": model_service.loaded,
        "repo_id": settings.HF_REPO_ID,
    }


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), language: str = "english"):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    try:
        _, tensor = preprocess_image(file.file, img_size=224)
        predictions = model_service.predict(tensor)
        
        # Create prediction objects
        prediction_objects = []
        for label, confidence in predictions:
            crop = label.split("_")[0]
            healthy = "healthy" in label
            friendly_name = model_service.get_friendly_disease_name(label, language)
            prediction_objects.append({
                "disease_name": friendly_name,  # Only friendly disease name
                "confidence": confidence,
                "crop": crop,
                "healthy": healthy
            })
        
        # Get advice for the top prediction
        top_label = predictions[0][0]
        advice = get_advice(top_label, language)

        note = None
        if predictions[0][1] < settings.CONFIDENCE_THRESHOLD:
            note = "Low confidence prediction. Try a clearer leaf photo in good daylight."

        return {
            "predictions": prediction_objects,
            "advice": advice,
            "note": note,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))