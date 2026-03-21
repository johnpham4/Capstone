<=
from pydantic import BaseModel


class Document(BaseModel):

    caption: str
    image_dir: str
    caption_vn: str

