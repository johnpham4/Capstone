from sqlalchemy.ext.asyncio import AsyncSession

from src.repositories.registry import RegistryRepository
from src.models.orm import RegistryModel
from src.models.dto import RegistryDTO

class RegistryService:
    def __init__(self, db: AsyncSession) -> None:
        self._res_repo = RegistryRepository

    def registry(self, request: RegistryDTO) -> RegistryModel:
        pass