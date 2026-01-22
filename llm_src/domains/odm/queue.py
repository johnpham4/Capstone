import json
from abc import ABC
from typing import Generic, Type, TypeVar, Callable, Dict, Any, Optional
from loguru import logger
from pydantic import BaseModel, Field
import pika

from llm_src.domains.exceptions import ImproperlyConfigured
from llm_src.infrastructures.messaging.rabbitmq import connection
from llm_src.settings import settings

T = TypeVar("T", bound="QueueBaseDocument")


class QueueBaseDocument(BaseModel, Generic[T], ABC):
    def to_queue(self: T, **kwargs) -> dict:
        """Convert model to queue message format."""
        return self.model_dump(**kwargs)

    @classmethod
    def from_queue(cls: Type[T], message: dict) -> T:
        if not message:
            raise ValueError("Message is empty")
        return cls(**message)

    def publish(self: T, queue_name: Optional[str] = None, **kwargs) -> bool:
        try:
            target_queue = queue_name or self.get_queue_name()
            message = self.to_queue(**kwargs)

            channel = connection.channel
            channel.queue_declare(queue=target_queue, durable=True)
            channel.basic_publish(
                exchange='',
                routing_key=target_queue,
                body=json.dumps(message),
                properties=pika.BasicProperties(delivery_mode=2)
            )

            logger.info(f"Published message to queue '{target_queue}'")
            return True

        except Exception as e:
            logger.exception(f"Failed to publish message: {e}")
            return False

    @classmethod
    def consume(
        cls: Type[T],
        callback: Callable,
        queue_name: Optional[str] = None,
        auto_ack: bool = False
    ) -> None:
        try:
            target_queue = queue_name or cls.get_queue_name()
            channel = connection.channel
            channel.queue_declare(queue=target_queue, durable=True)

            def wrapped_callback(ch, method, properties, body):
                try:
                    message = json.loads(body)
                    instance = cls.from_queue(message)
                    callback(ch, method, properties, instance)
                except Exception as e:
                    logger.exception(f"Error processing message: {e}")
                    if not auto_ack:
                        ch.basic_ack(delivery_tag=method.delivery_tag)

            channel.basic_consume(
                queue=target_queue,
                on_message_callback=wrapped_callback,
                auto_ack=auto_ack
            )

            logger.info(f"Started consuming from queue '{target_queue}'")
            channel.start_consuming()

        except KeyboardInterrupt:
            logger.info("Consumer stopped by user")
            channel.stop_consuming()
        except Exception as e:
            logger.exception(f"Error in consumer: {e}")
            raise

    @classmethod
    def get_queue_name(cls: Type[T]) -> str:
        """Get the queue name for this document type."""
        if not hasattr(cls, "Settings") or not hasattr(cls.Settings, "queue_name"):
            raise ImproperlyConfigured(
                "Document should define a Settings configuration class with the queue_name attribute."
            )
        return cls.Settings.queue_name
