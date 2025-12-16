from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")

    # MongoDB database
    DATABASE_HOST: str = "mongodb://geo_engineering:geo_engineering@127.0.0.1:27017"
    DATABASE_NAME: str = "unigeo"

    # Model paths
    VQ_MODEL: str = "JO-KU/Geo-MAGVIT"
    BASE_LLM: str = "Qwen/Qwen2.5-3B-Instruct"

    YOLOV_SEGMENTATION_MODEL_ID: str = "yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt"

    # Figure Extraction
    FIGURE_DETECTOR_MODEL: str = "juliozhao/DocLayout-YOLO-DocStructBench-imgsz1280-2501"
    FIGURE_DETECTOR_LOCAL_DIR: str = "./models/DocLayout-YOLO-DocStructBench-imgsz1280-2501"
    FIGURE_DETECTOR_DEVICE: str = "auto"
    FIGURE_DETECTION_CONF: float = 0.25
    FIGURE_DETECTION_IMGSZ: int = 1280

    # Training
    OUTPUT_DIR: str = "./checkpoints"
    BATCH_SIZE: int = 1
    EPOCHS: int = 3
    LEARNING_RATE: float = 2e-4

    HF_TOKEN: str


settings = Settings()