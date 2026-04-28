"""History schemas — giữ flat, giữ ít, mỗi class có đúng 1 mục đích."""

from datetime import datetime

from pydantic import BaseModel


class HistoryItem(BaseModel):
    id: str
    input_text: str
    mode: str
    status: str
    latency_ms: int | None = None
    has_diagram: bool = False
    has_solution: bool = False
    has_source_image: bool = False
    has_ocr: bool = False
    created_at: datetime

    model_config = {"from_attributes": True}


class HistoryDetail(HistoryItem):
    updated_at: datetime
    source_image_url: str | None = None
    ocr_text: str | None = None
    dsl: str | None = None
    image_url: str | None = None
    image_base64: str | None = None
    solution: str | None = None


class PaginatedHistory(BaseModel):
    items: list[HistoryItem]
    total: int
    page: int
    page_size: int
