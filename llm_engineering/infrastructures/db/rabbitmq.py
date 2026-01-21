import json
<<<<<<< HEAD
from pika import BlockingConnection, URLParameters
=======
import pika
>>>>>>> quang
from typing import Callable, Dict, Any, Optional
from loguru import logger
from abc import ABC, abstractmethod

from llm_engineering.settings import settings


<<<<<<< HEAD
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
=======
class RabbitMQConnection:
    """Base RabbitMQ connection manager"""

    def __init__(self, url: Optional[str] = None):
        self.url = url or settings.RABBITMQ_URL
        self.connection = None
        self.channel = None

    def connect(self):
        """Establish connection to RabbitMQ"""
        try:
            self.connection = pika.BlockingConnection(pika.URLParameters(self.url))
            self.channel = self.connection.channel()
            logger.info("Connected to RabbitMQ")
        except Exception as e:
            logger.exception(f"Failed to connect to RabbitMQ: {e}")
            raise

    def close(self):
        """Close RabbitMQ connection"""
        if self.connection and not self.connection.is_closed:
            self.connection.close()
            logger.info("RabbitMQ connection closed")


class RabbitMQPublisher(RabbitMQConnection):
    """Publisher for sending messages to RabbitMQ queues"""

    def declare_queue(self, queue_name: str, durable: bool = True):
        """Declare a queue"""
        if not self.channel:
            self.connect()
        self.channel.queue_declare(queue=queue_name, durable=durable)
        logger.info(f"Queue '{queue_name}' declared")

    def publish(self, queue_name: str, message: Dict[str, Any]):
        """Publish a message to a queue"""
        try:
            if not self.channel:
                self.connect()

            self.channel.basic_publish(
                exchange='',
                routing_key=queue_name,
                body=json.dumps(message),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # make message persistent
                )
            )
            logger.info(f"Message published to queue '{queue_name}': {message.get('request_id', 'N/A')}")
        except Exception as e:
            logger.exception(f"Failed to publish message to '{queue_name}': {e}")
            raise


class RabbitMQConsumer(RabbitMQConnection):
    """Base consumer for receiving messages from RabbitMQ queues"""

    def __init__(self, queue_name: str, callback: Callable, url: Optional[str] = None):
        super().__init__(url)
        self.queue_name = queue_name
        self.callback = callback

    def declare_queue(self, durable: bool = True):
        """Declare the queue"""
        if not self.channel:
            self.connect()
        self.channel.queue_declare(queue=self.queue_name, durable=durable)
        logger.info(f"Queue '{self.queue_name}' declared for consumer")

    def _on_message_callback(self, ch, method, properties, body):
        """Internal callback wrapper"""
        try:
            message = json.loads(body)
            logger.info(f"Received message from '{self.queue_name}': {message.get('request_id', 'N/A')}")
            self.callback(ch, method, properties, message)
        except Exception as e:
            logger.exception(f"Error processing message: {e}")
            # Acknowledge to remove from queue even on error (or implement retry logic)
            ch.basic_ack(delivery_tag=method.delivery_tag)

    def start_consuming(self):
        """Start consuming messages"""
        try:
            if not self.channel:
                self.connect()

            self.declare_queue()
            self.channel.basic_consume(
                queue=self.queue_name,
                on_message_callback=self._on_message_callback,
                auto_ack=False
            )

            logger.info(f"Started consuming from queue '{self.queue_name}'")
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("Consumer stopped by user")
            self.stop_consuming()
        except Exception as e:
            logger.exception(f"Error in consumer: {e}")
            raise

    def stop_consuming(self):
        """Stop consuming messages"""
        if self.channel:
            self.channel.stop_consuming()
        self.close()
>>>>>>> quang
