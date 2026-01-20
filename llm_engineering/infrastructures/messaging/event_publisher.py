"""
Event Publisher for RabbitMQ messaging.

Handles publishing domain events to RabbitMQ queues.
"""
import json
from typing import Optional
from loguru import logger
import pika

from llm_engineering.domains.events import Event
from llm_engineering.infrastructures.messaging.rabbitmq import connection


class EventPublisher:

    def __init__(self):
        self.connection = connection

    def publish(self, event: Event, queue_name: str, **kwargs) -> bool:

        try:
            # Convert event to dict
            message = event.model_dump(**kwargs)

            # Declare queue (idempotent)
            channel = self.connection.channel
            channel.queue_declare(queue=queue_name, durable=True)

            # Publish message
            channel.basic_publish(
                exchange='',
                routing_key=queue_name,
                body=json.dumps(message),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Make message persistent
                    content_type='application/json'
                )
            )

            logger.info(f"Published {event.event_type} to queue '{queue_name}' (event_id: {event.event_id})")
            return True

        except Exception as e:
            logger.exception(f"Failed to publish event to {queue_name}: {e}")
            return False

    def publish_batch(self, events: list[Event], queue_name: str) -> int:
        """
        Publish multiple events to a queue.

        Returns:
            Number of successfully published events
        """
        success_count = 0
        for event in events:
            if self.publish(event, queue_name):
                success_count += 1

        logger.info(f"Published {success_count}/{len(events)} events to {queue_name}")
        return success_count
