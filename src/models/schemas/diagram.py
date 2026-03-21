from pydantic import BaseModel, Field
from typing import Optional, Literal


class DiagramRequest(BaseModel):
    problem: str = Field(..., description="Natural language geometry problem description")
    language: Literal["vi", "en"] = Field(default="vi", description="Problem language")


class DiagramResponse(BaseModel):
    request_id: str
    dsl: str
    diagram_url: Optional[str] = None
    error: Optional[str] = None


class RenderRequest(BaseModel):
    dsl: str = Field(..., description="Geometry DSL code")
    format: Literal["png", "svg", "pdf"] = Field(default="png", description="Output format")
    width: int = Field(default=800, ge=100, le=2000, description="Image width in pixels")
    height: int = Field(default=600, ge=100, le=2000, description="Image height in pixels")
    title: Optional[str] = Field(None, description="Diagram title")
    epochs: int = Field(default=1000, ge=100, le=5000, description="Optimization epochs")
    n_tries: int = Field(default=1, ge=1, le=10, description="Number of optimization attempts")
    dpi: int = Field(default=150, ge=72, le=300, description="Image DPI")


class RenderResponse(BaseModel):
    """Response schema for diagram rendering."""
    request_id: str
    format: str
    success: bool
    error: Optional[str] = None

