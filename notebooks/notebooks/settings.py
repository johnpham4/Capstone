from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
	model_config = SettingsConfigDict(
		env_file=Path(__file__).resolve().parents[1] / ".env",
		extra="ignore",
	)

	openai_api_key: str | None = None

settings = Settings()
