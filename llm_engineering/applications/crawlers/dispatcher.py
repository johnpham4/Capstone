import re
from urllib.parse import urlparse
from loguru import logger
from .base import BaseCrawler
from .pdf import PDFCrawler

class CrawlerDispatcher:
    def __init__(self):
        self._crawlers = {}

    @classmethod
    def build(cls) -> "CrawlerDispatcher":
        return cls()

    def register_pdf(self) -> "CrawlerDispatcher":
        self.register("pdf", PDFCrawler())
        return self

    def register(self, source_type: str, crawler: BaseCrawler) -> None:
        self._crawlers[source_type] = crawler

    def get_crawler(self, url: str) -> BaseCrawler:
        parsed = urlparse(url)
        path = parsed.path.lower()

        if path.endswith('.pdf'):
            return self._crawlers.get('pdf')
        elif path.endswith(('.doc', '.docx')):
            return self._crawlers.get('docx')

        logger.warning(f"No crawler found for {url}")
        return None


