from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")

    # MongoDB database
    DATABASE_HOST: str = "mongodb://geo_engineering:geo_engineering@127.0.0.1:27017"
    DATABASE_NAME: str = "unigeo"

    BASE_LLM: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    HF_TOKEN: str | None = None

    # AWS Credentials
    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_REGION: str = "us-east-1"
    AWS_ARN_ROLE: str | None = None

    # HuggingFace Model
    HF_MODEL_ID: str = "minn4/GeoUni-Qwen2.5-7B"
    HUGGINGFACE_ACCESS_TOKEN: str | None = None

    # SageMaker Endpoint Config
    SAGEMAKER_ENDPOINT_INFERENCE: str = "text2diagram-llm-endpoint"
    SAGEMAKER_ENDPOINT_CONFIG_INFERENCE: str = "text2diagram-llm-endpoint"
    GPU_INSTANCE_TYPE: str = "ml.g5.2xlarge"

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