import base64
import io
import os
import time
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from groq import Groq
from PIL import Image, ImageOps
from pydantic import BaseModel

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY chưa được thiết lập trong file .env")

# Sử dụng model Qwen 3.8 27B khả dụng trên tài khoản Groq của bạn
MODEL_NAME = "qwen/qwen3.8-27b"
groq_client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="Smart Clinic VLM Playground")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatResponse(BaseModel):
    model: str
    latency_ms: float
    reply: str

def optimize_image(image_bytes: bytes, max_dimension: int = 1200, quality: int = 80) -> tuple[str, str]:
    """Tự động xoay đúng EXIF, thu nhỏ kích thước và nén JPEG trong RAM để giảm tải network."""
    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)  # Giữ đúng chiều xoay gốc của camera

    # Chuyển đổi RGB nếu là ảnh RGBA/PNG
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    # Thu nhỏ nếu chiều dài nhất vượt quá max_dimension
    if max(image.size) > max_dimension:
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    optimized_bytes = buffer.getvalue()
    
    return base64.b64encode(optimized_bytes).decode("utf-8"), "image/jpeg"

@app.get("/")
async def serve_playground():
    if not os.path.exists("index.html"):
        raise HTTPException(status_code=404, detail="Không tìm thấy file index.html")
    return FileResponse("index.html")

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(
    prompt: str = Form(...),
    file: Optional[UploadFile] = File(None),
):
    start_time = time.perf_counter()

    try:
        content_payload = []

        if file and file.filename:
            raw_bytes = await file.read()
            # Nén ảnh xuống ~150KB trước khi gửi qua API
            base64_img, mime_type = optimize_image(raw_bytes)
            
            content_payload.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_img}"}
            })

        system_instruction = (
            "Bạn là chuyên gia OCR bóc tách thông tin thẻ Bảo hiểm Y tế (BHYT) Việt Nam.\n"
            "Ảnh có thể bị xoay 15-40 độ hoặc bị đảo ngược. Hãy tự căn chỉnh và trích xuất.\n"
            "Chỉ trả về DUY NHẤT một JSON object theo đúng schema (không kèm lời dẫn giải thích):\n"
            "{\n"
            '  "ma_so_bhyt": "15 ký tự mã thẻ",\n'
            '  "ho_ten": "Họ và tên in hoa",\n'
            '  "ngay_sinh": "DD/MM/YYYY",\n'
            '  "gioi_tinh": "Nam hoặc Nữ",\n'
            '  "noi_kham_chua_benh_ban_dau": "string",\n'
            '  "gia_tri_su_dung_tu": "DD/MM/YYYY",\n'
            '  "gia_tri_su_dung_den": "DD/MM/YYYY hoặc null",\n'
            '  "con_han": true,\n'
            '  "ghi_chu": "string hoặc null"\n'
            "}"
        )

        content_payload.insert(0, {"type": "text", "text": f"{system_instruction}\n\nYêu cầu: {prompt}"})

        # Giới hạn max_completion_tokens để model không suy luận lan man
        chat_completion = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": content_payload}],
            temperature=0.1,
            max_completion_tokens=512,
            response_format={"type": "json_object"},
        )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        reply_content = chat_completion.choices[0].message.content

        return ChatResponse(
            model=MODEL_NAME,
            latency_ms=elapsed_ms,
            reply=reply_content
        )

    except Exception as e:
        print(f"===> LỖI GROQ API: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    finally:
        if file:
            await file.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


# ## Gemini model
# import os
# import time
# from typing import Optional
# from dotenv import load_dotenv
# from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse
# from google import genai
# from google.genai import types
# from pydantic import BaseModel

# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# if not GEMINI_API_KEY:
#     raise RuntimeError("GEMINI_API_KEY chưa được thiết lập trong file .env")

# # Model thị giác đa phương thức tốc độ cao của Google
# MODEL_NAME = "gemini-3.6-flash"
# client = genai.Client(api_key=GEMINI_API_KEY)

# app = FastAPI(title="Smart Clinic VLM Playground")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class ChatResponse(BaseModel):
#     model: str
#     latency_ms: float
#     reply: str

# @app.get("/")
# async def serve_playground():
#     if not os.path.exists("index.html"):
#         raise HTTPException(status_code=404, detail="Không tìm thấy file index.html")
#     return FileResponse("index.html")

# @app.post("/api/v1/chat", response_model=ChatResponse)
# async def chat_endpoint(
#     prompt: str = Form(...),
#     file: Optional[UploadFile] = File(None),
# ):
#     start_time = time.perf_counter()

#     try:
#         contents_payload = []

#         # Đọc dữ liệu ảnh từ buffer bộ nhớ
#         if file and file.filename:
#             image_bytes = await file.read()
#             mime_type = file.content_type or "image/jpeg"
#             contents_payload.append(
#                 types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
#             )

#         system_instruction = (
#             "Bạn là trợ lý AI chuyên bóc tách thông tin thẻ Bảo hiểm Y tế (BHYT) Việt Nam.\n"
#             "Ảnh có thể bị nghiêng từ 15-40 độ hoặc bị đảo ngược. Hãy tự căn chỉnh và trích xuất.\n"
#             "Bắt buộc trả về DUY NHẤT một chuỗi JSON chuẩn (không dùng cú pháp markdown ```json ```) theo cấu trúc sau:\n"
#             "{\n"
#             '  "ma_so_bhyt": "15 ký tự mã thẻ",\n'
#             '  "ho_ten": "Họ và tên in hoa",\n'
#             '  "ngay_sinh": "DD/MM/YYYY",\n'
#             '  "gioi_tinh": "Nam hoặc Nữ",\n'
#             '  "noi_kham_chua_benh_ban_dau": "string",\n'
#             '  "gia_tri_su_dung_tu": "DD/MM/YYYY",\n'
#             '  "gia_tri_su_dung_den": "DD/MM/YYYY hoặc null",\n'
#             '  "con_han": true,\n'
#             '  "ghi_chu": "string hoặc null"\n'
#             "}"
#         )

#         contents_payload.append(f"{system_instruction}\n\nYêu cầu bổ sung: {prompt}")

#         # Gọi mô hình Gemini
#         response = client.models.generate_content(
#             model=MODEL_NAME,
#             contents=contents_payload,
#         )

#         elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
#         reply_text = response.text or "Không nhận được phản hồi từ mô hình."

#         return ChatResponse(
#             model=MODEL_NAME,
#             latency_ms=elapsed_ms,
#             reply=reply_text.strip()
#         )

#     except Exception as e:
#         print(f"===> LỖI GEMINI API: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=str(e)
#         )
#     finally:
#         if file:
#             await file.close()

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)