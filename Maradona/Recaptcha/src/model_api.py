from fastapi import FastAPI, UploadFile, File
from inference import RecaptchaPredictor
from io import BytesIO
from PIL import Image
import uvicorn
import os

app = FastAPI(title="Recaptcha Classification API", description="V3 EfficientNet-B0 Model Inference API")

# 모델 경로 설정
MODEL_PATH = "models/v3/cnn_best_model.pth"

import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

# 전역 예측기 변수 (앱 시작 시 로드)
predictor = None

logger.info(f"📂 Current Working Directory: {os.getcwd()}")
logger.info(f"📂 Expected Model Path: {os.path.abspath(MODEL_PATH)}")

if os.path.exists(MODEL_PATH):
    try:
        predictor = RecaptchaPredictor(MODEL_PATH)
        logger.info("✅ Model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        import traceback
        logger.error(traceback.format_exc())
else:
    logger.error(f"⚠️ Model file not found at {MODEL_PATH}. API will return errors.")

# @app.on_event("startup") 제거 (전역에서 처리됨)


@app.get("/")
def health_check():
    """서버 상태 확인용 엔드포인트"""
    status = "healthy" if predictor is not None else "unhealthy"
    return {"status": status, "model_version": "v3", "architecture": "EfficientNet-B0"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """이미지 파일을 업로드받아 클래스를 예측합니다."""
    if predictor is None:
        return {"error": "Model is not loaded properly."}
    
    try:
        # 업로드된 파일 읽기
        content = await file.read()
        image = Image.open(BytesIO(content))
        
        # 추론 수행
        result = predictor.predict(image)
        return result
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
