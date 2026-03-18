from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.predict import router as predict_router
from app.services.model_service import model_service

app = FastAPI(
    title="FarmGuard AI API",
    description="AI-powered crop disease detection API for cassava, maize, tomato, and pepper.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api", tags=["Prediction"])


@app.on_event("startup")
def startup_event():
    model_service.load()


@app.get("/")
def root():
    return {"message": "FarmGuard AI backend is running"}