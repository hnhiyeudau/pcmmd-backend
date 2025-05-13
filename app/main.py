from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from app.service.predict import predict_image

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@app.get("/")
def root():
    return {"message": "PCMMD FastAPI backend is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    result = await predict_image(file)
    return result
