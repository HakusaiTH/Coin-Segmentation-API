from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
import requests
import base64
from io import BytesIO

# สร้างแอป FastAPI
app = FastAPI()

# สร้างโมเดลสำหรับรับข้อมูล JSON
class ImageURL(BaseModel):
    url: str

# ฟังก์ชันสำหรับดาวน์โหลดรูปภาพจาก URL
def download_image(url: str):
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to download image")
    img_array = np.frombuffer(response.content, np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# ฟังก์ชันสำหรับแปลงภาพกลับเป็น Base64
def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text

# เส้นทาง API หลัก
@app.post("/process-image/")
async def process_image(data: ImageURL):
    try:
        # ดาวน์โหลดรูปภาพ
        image = download_image(data.url)

        # ประมวลผลภาพ
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
        thresh = cv2.adaptiveThreshold(
            gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)
        result_img = closing.copy()
        contours, _ = cv2.findContours(
            result_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        counter = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 5000 < area < 35000:
                ellipse = cv2.fitEllipse(cnt)
                cv2.ellipse(image, ellipse, (0, 255, 0), 2)
                counter += 1

        # วาดจำนวนวัตถุในภาพ
        cv2.putText(image, str(counter), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 2, cv2.LINE_AA)

        # แปลงภาพที่ประมวลผลแล้วกลับเป็น Base64
        processed_image_base64 = image_to_base64(image)

        # ส่งผลลัพธ์กลับไป
        return {"object_count": counter, "processed_image": processed_image_base64}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
