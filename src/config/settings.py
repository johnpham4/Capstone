from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")

    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "capstone"

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # HuggingFace Model
    BASE_LLM: str = "unsloth/Falcon-unsloth/Qwen2.5-7B-Instruct-7B"
    HF_MODEL_ID: str = f"minn4/text2diagram-Qwen2.5-7B-Instruct"
    HF_TOKEN: str | None = None

    # AWS Credentials
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_REGION: str = "us-east-1"
    REGION_NAME: str | None = None
    AWS_ARN_ROLE: str | None = None
    S3_BUCKET_NAME: str | None = None
    S3_DIAGRAM_PREFIX: str = "diagrams"

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

    APP_ENV: str = "development"
    JWT_SECRET_KEY: str = "change-me-in-env"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    CORS_ALLOW_ORIGINS: str = "*"
    CORS_ALLOW_CREDENTIALS: bool = False

    INIT_DB_ON_STARTUP: bool = True

    # RabbitMQ Configuration
    RABBITMQ_URL: str = "amqp://guest:guest@localhost:5672/"

    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379/0"

    # Diagram Generation
    OUTPUT_DIR: str = "./output/diagrams"
    DIAGRAM_OPTIMIZER_EPOCHS: int = 3000
    DIAGRAM_OPTIMIZER_LR: float = 0.01
    DIAGRAM_QUEUE_NAME: str = "diagram.render"
    DIAGRAM_QUEUE_EXCHANGE: str = "diagram"
    DIAGRAM_QUEUE_ROUTING_KEY: str = "diagram.render"

    # LLM Endpoint for DSL generation (ngrok / SageMaker / local)
    LLM_ENDPOINT_URL: str | None = None
    LLM_ENDPOINT_TIMEOUT: int = 60

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

    @property
    def cors_origins(self) -> list[str]:
        raw = (self.CORS_ALLOW_ORIGINS or "").strip()
        if not raw or raw == "*":
            return ["*"]
        return [origin.strip() for origin in raw.split(",") if origin.strip()]

settings = Settings()