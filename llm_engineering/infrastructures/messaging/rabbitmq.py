import json
from pika import BlockingConnection, URLParameters
from typing import Callable, Dict, Any, Optional
from loguru import logger
from abc import ABC, abstractmethod

from llm_engineering.settings import settings


class RabbitMQConnector:
    _connection: Optional[BlockingConnection] = None
    _channel: Any | None = None

    @classmethod
    def __new__(cls, *args, **kwargs):
        if cls._connection is None or cls._connection.is_closed:
            try:
                cls._connection = BlockingConnection(
                    URLParameters(settings.RABBITMQ_URL)
                )
                cls._channel = cls._connection.channel()
                logger.info("Connected to RabbitMQ")
            except Exception as e:
                logger.exception(f"Failed to connect to RabbitMQ: {e}")
                raise

        return cls._channel

    @classmethod
    def close(cls):
        if cls._connection and not cls._connection.is_closed:
            cls._connection.close()
            cls._connection = None
            cls._channel = None
            logger.info("RabbitMQ connection closed")

connection = RabbitMQConnector()