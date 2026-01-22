from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")

    # MongoDB database
    DATABASE_HOST: str = "mongodb://geo_engineering:geo_engineering@127.0.0.1:27017"
    DATABASE_NAME: str = "unigeo"

    # HuggingFace Model
    BASE_LLM: str = "unsloth/Falcon-H1R-7B"
    HF_MODEL_ID: str = f"minn4/text2diagram-Falcon-H1R-7B"
    HF_TOKEN: str | None = None

    # AWS Credentials
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_REGION: str = "us-east-1"
    AWS_ARN_ROLE: str | None = None

    # SageMaker Endpoint Config
    SAGEMAKER_ENDPOINT_INFERENCE: str = "text2diagram-llm-endpoint"
    SAGEMAKER_ENDPOINT_CONFIG_INFERENCE: str = "text2diagram-llm-endpoint"
    GPU_INSTANCE_TYPE: str = "ml.g5.2xlarge"

    # KV Cache & Optimization
    USE_CACHE_INFERENCE: bool = True  # Enable KV cache for faster inference
    MAX_CONCURRENT_REQUESTS: int = 128  # Max concurrent requests
    MAX_WAITING_TOKENS: int = 20  # Max tokens waiting in queue

    # Model Inference Config
    SM_NUM_GPUS: int = 1
    MAX_INPUT_LENGTH: int = 1024
    MAX_TOTAL_TOKENS: int = 2048
    MAX_BATCH_TOTAL_TOKENS: int = 4096
    MAX_BATCH_PREFILL_TOKENS: int = 4096
    MAX_NEW_TOKENS_INFERENCE: int = 512
    TOP_P_INFERENCE: float = 0.9
    TEMPERATURE_INFERENCE: float = 0.7

    COPIES: int = 1
    GPUS: int = 1
    CPUS: int = 4

    # Comet ML
    COMET_API_KEY: str | None = None
    COMET_PROJECT: str = "geouni-finetuning"

    OPENAI_MODEL_ID: str = "gpt-4o-mini"
    OPENAI_API_KEY: str | None = None

    # RabbitMQ Configuration
    RABBITMQ_URL: str = "amqp://guest:guest@localhost:5672/"

    # Diagram Generation
    OUTPUT_DIR: str = "./output/diagrams"
    DIAGRAM_OPTIMIZER_EPOCHS: int = 1000
    DIAGRAM_OPTIMIZER_LR: float = 0.01

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