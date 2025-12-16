from abc import ABC, abstractmethod


class BaseCrawler(ABC):
    @abstractmethod
    def extract(self, link: str) -> bytes:
        pass
