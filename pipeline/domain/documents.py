<<<<<<<< HEAD:src/models/domain/training/documents.py
from pydantic import BaseModel


class Document(BaseModel):

    caption: str
    image_dir: str
    caption_vn: str
========
from pydantic import BaseModel


class Document(BaseModel):

    caption: str
    image_dir: str
    caption_vn: str
>>>>>>>> 6cf03dda8dad8bb8fa1226b8b4e9166c3f287527:pipeline/domain/documents.py
