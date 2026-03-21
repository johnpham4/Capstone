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
>>>>>>>> minh-re:pipeline/domain/documents.py
