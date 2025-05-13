# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from app.service.predict import predict_image

# app = FastAPI()

# # origins = [
# #     "https://pcmmd-frontend-mv2w5lz5s-nhutduys-projects.vercel.app",
# #     "http://localhost:3000",  # Cho phép dùng local test luôn
# # ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["https://pcmmd-frontend-mv2w5lz5s-nhutduys-projects.vercel.app",
#     "http://localhost:3000",
#     ], 
#     allow_credentials=True,
#     allow_methods=["*"], 
#     allow_headers=["*"]
# )

# @app.get("/")
# def root():
#     return {"message": "PCMMD FastAPI backend is running!"}

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     result = await predict_image(file)
#     return result




from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from app.service.predict import predict_image

app = FastAPI()

# Cho phép tất cả origin gọi API (phù hợp nếu frontend trên Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Hoặc chỉ frontend của bạn: ["https://pcmmd-frontend.vercel.app"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "PCMMD FastAPI backend is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Nhận ảnh từ client, dự đoán bằng YOLO, trả về nhãn và độ tin cậy.
    """
    result = await predict_image(file)
    return result
