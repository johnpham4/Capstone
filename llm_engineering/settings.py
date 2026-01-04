from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")

    # MongoDB database
    DATABASE_HOST: str = "mongodb://geo_engineering:geo_engineering@127.0.0.1:27017"
    DATABASE_NAME: str = "unigeo"

    BASE_LLM: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

    YOLOV_SEGMENTATION_MODEL_ID: str = "yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt"

    # Figure Extraction
    FIGURE_DETECTOR_MODEL: str = "juliozhao/DocLayout-YOLO-DocStructBench-imgsz1280-2501"
    FIGURE_DETECTOR_LOCAL_DIR: str = "./models/DocLayout-YOLO-DocStructBench-imgsz1280-2501"
    FIGURE_DETECTOR_DEVICE: str = "auto"
    FIGURE_DETECTION_CONF: float = 0.25
    FIGURE_DETECTION_IMGSZ: int = 1280

    HF_TOKEN: str

    OPENAI_MODEL_ID: str = "gpt-4o-mini"
    OPENAI_API_KEY: str | None = None

    @property
    def OPENAI_MAX_TOKEN_WINDOW(self) -> int:
        official_max_token_window = {
            "gpt-3.5-turbo": 16385,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
        }.get(self.OPENAI_MODEL_ID, 128000)

        max_token_window = int(official_max_token_window * 0.90)

        return max_token_window

settings = Settings()