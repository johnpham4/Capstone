"""API schemas for diagram endpoints."""

from pydantic import BaseModel, Field
from typing import Optional, Literal


class DiagramRequest(BaseModel):
    """Request schema for diagram generation from natural language."""
    problem: str = Field(..., description="Natural language geometry problem description")
    language: Literal["vi", "en"] = Field(default="vi", description="Problem language")


class DiagramResponse(BaseModel):
    """Response schema for diagram generation."""
    request_id: str
    dsl: str
    diagram_url: Optional[str] = None
    error: Optional[str] = None


class RenderRequest(BaseModel):
    """Request schema for rendering diagram from DSL."""
    dsl: str = Field(..., description="Geometry DSL code")
    format: Literal["png", "svg", "pdf"] = Field(default="png", description="Output format")
    width: int = Field(default=800, ge=100, le=2000, description="Image width in pixels")
    height: int = Field(default=600, ge=100, le=2000, description="Image height in pixels")
    title: Optional[str] = Field(None, description="Diagram title")


class RenderResponse(BaseModel):
    """Response schema for diagram rendering."""
    request_id: str
    format: str
    success: bool
    error: Optional[str] = None
