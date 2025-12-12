from abc import ABC, abstractmethod
from pydantic import BaseModel
from bs4 import BeatifulSoup

class BaseCrawler(BaseModel, ABC):
    
    @abstractmethod
    def extract(self, link: str, **kwargs) -> None: ...