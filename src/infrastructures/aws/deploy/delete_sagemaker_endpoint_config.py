<<<<<<< HEAD
import boto3
from loguru import logger
from src.config.settings.base import settings


def delete_endpoint_config():
    client = boto3.client(
        'sagemaker',
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )

    config_name = settings.SAGEMAKER_ENDPOINT_CONFIG_INFERENCE

    try:
        response = client.describe_endpoint_config(EndpointConfigName=config_name)
        logger.info(f"Found endpoint config: {config_name}")

        client.delete_endpoint_config(EndpointConfigName=config_name)
        logger.success(f"Deleted endpoint config: {config_name}")

    except client.exceptions.ResourceNotFound:
        logger.warning(f"Endpoint config '{config_name}' không tồn tại")
    except Exception as e:
        logger.warning(f"Error: {e}")


if __name__ == "__main__":
=======
import boto3
from loguru import logger
from src.config.settings.base import settings


def delete_endpoint_config():
    client = boto3.client(
        'sagemaker',
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )

    config_name = settings.SAGEMAKER_ENDPOINT_CONFIG_INFERENCE

    try:
        response = client.describe_endpoint_config(EndpointConfigName=config_name)
        logger.info(f"Found endpoint config: {config_name}")

        client.delete_endpoint_config(EndpointConfigName=config_name)
        logger.success(f"Deleted endpoint config: {config_name}")

    except client.exceptions.ResourceNotFound:
        logger.warning(f"Endpoint config '{config_name}' không tồn tại")
    except Exception as e:
        logger.warning(f"Error: {e}")


if __name__ == "__main__":
>>>>>>> minh-re
    delete_endpoint_config()