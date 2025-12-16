"""Pipeline for extracting figures and text from PDFs."""

from typing import List, Dict, Any
from zenml import pipeline

from steps.crawl.crawl_data import crawl_links


@pipeline
def figure_extraction_pipeline(pdf_sources: List[Dict[str, Any]]):
    """
    Extract figures and text problems from PDF sources in one pass.
    - Extract figures with captions
    - Extract text problems with figure references
    - Map problems to figures automatically
    """
    crawl_links(pdf_sources=pdf_sources)
