from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import joinedload

from src.repositories.base import AbstractRepository
from src.models.orm import RegistryModel


class RegistryRepository(AbstractRepository[RegistryModel]):

    model = RegistryModel
