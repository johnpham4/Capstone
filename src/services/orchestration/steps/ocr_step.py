from openai import OpenAI
from loguru import logger

from src.config.settings import settings

OCR_SYSTEM_PROMPT = (
    "Bạn là công cụ OCR chuyên đọc đề bài toán hình học từ ảnh.\n"
    "Nhiệm vụ: Trích xuất CHÍNH XÁC toàn bộ nội dung đề bài hình học trong ảnh, "
    "bao gồm giả thiết, yêu cầu chứng minh/tính toán, và các số liệu.\n"
    "Quy tắc:\n"
    "- Giữ nguyên ngôn ngữ gốc (tiếng Việt hoặc tiếng Anh).\n"
    "- Giữ nguyên ký hiệu toán học (AB, ∠, °, ⊥, //, △, ...).\n"
    "- Nếu ảnh có hình vẽ kèm đề, mô tả ngắn gọn hình vẽ ở cuối.\n"
    "- CHỈ trả về nội dung đề bài, KHÔNG thêm lời giải hay nhận xét.\n"
    "- Nếu không đọc được hoặc ảnh không chứa đề toán, trả về chuỗi rỗng."
)


class OcrStep:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=30.0,
        )

    def execute(self, image_base64: str, hint: str = "") -> dict:
        """Extract geometry problem text from an image using GPT-4o vision."""
        try:
            extracted = self._extract_text(image_base64, hint)
            if not extracted.strip():
                return {"extracted_text": "", "status": "failed", "error": "Could not extract text from image"}

            logger.info(f"OcrStep extracted {len(extracted)} chars")
            return {"extracted_text": extracted, "status": "success"}
        except Exception as e:
            logger.error(f"OcrStep error: {e}")
            return {"extracted_text": "", "status": "failed", "error": str(e)}

    def _extract_text(self, image_base64: str, hint: str) -> str:
        user_content: list[dict] = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}", "detail": "high"},
            },
        ]
        if hint:
            user_content.append({"type": "text", "text": f"Gợi ý thêm từ người dùng: {hint}"})

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": OCR_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=1024,
            temperature=0.1,
        )
        return (response.choices[0].message.content or "").strip()
