import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


SYSTEM_PROMPT = """
Bạn là hệ thống sửa lỗi OCR cho đề toán tiếng Việt.

NHIỆM VỤ:
- Sửa lỗi chính tả tiếng Việt do OCR
- Chuyển ký hiệu toán học thành chữ
- Không thêm hoặc bớt dữ kiện
- Không giải bài toán
- Giữ nguyên nhãn hình: H.1, (H.2.3)
"""

def fix_ocr_text(text: str) -> str:
    if not text.strip():
        return ""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ]
    )

    return response.choices[0].message.content.strip()
