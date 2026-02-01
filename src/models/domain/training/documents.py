"""Document domain models."""

from pydantic import BaseModel


class Document(BaseModel):
    """Document with caption and image.
    
    Uses BaseModel for validation and serialization.
    """
    caption: str
    image_dir: str
    caption_vn: str
