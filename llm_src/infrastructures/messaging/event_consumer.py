"""
Event Consumer for RabbitMQ messaging.

Handles consuming domain events from RabbitMQ queues.
"""
import json
from typing import Callable, Type
from loguru import logger

from llm_src.domains.events import Event
from llm_src.infrastructures.messaging.rabbitmq import connection


class EventConsumer:

    def __init__(self):
        self.connection = connection

    def consume(
        self,
        queue_name: str,
        event_class: Type[Event],
        callback: Callable[[Event], None],
        auto_ack: bool = False
    ) -> None:

        try:
            channel = self.connection.channel
            channel.queue_declare(queue=queue_name, durable=True)

            def wrapped_callback(ch, method, properties, body):
                try:
                    # Parse message
                    message = json.loads(body)
                    event = event_class(**message)

                    # Call user callback
                    callback(event)

                    # Acknowledge message if not auto_ack
                    if not auto_ack:
                        ch.basic_ack(delivery_tag=method.delivery_tag)

                    logger.debug(f"Processed {event.event_type} from {queue_name}")

                except Exception as e:
                    logger.exception(f"Error processing message from {queue_name}: {e}")
                    # Acknowledge to prevent infinite redelivery
                    if not auto_ack:
                        ch.basic_ack(delivery_tag=method.delivery_tag)

            # Set prefetch to 1 for fair dispatch
            channel.basic_qos(prefetch_count=1)

            # Start consuming
            channel.basic_consume(
                queue=queue_name,
                on_message_callback=wrapped_callback,
                auto_ack=auto_ack
            )

            logger.info(f"Started consuming from queue '{queue_name}'")
            channel.start_consuming()

        except KeyboardInterrupt:
            logger.info(f"Consumer stopped by user")
            channel.stop_consuming()
        except Exception as e:
            logger.exception(f"Error in consumer for {queue_name}: {e}")
            raise

    def consume_raw(
        self,
        queue_name: str,
        callback: Callable,
        auto_ack: bool = False
    ) -> None:
        """
        Consume raw messages (without deserialization).

        Callback receives (ch, method, properties, body) like native pika.
        """
        try:
            channel = self.connection.channel
            channel.queue_declare(queue=queue_name, durable=True)
            channel.basic_qos(prefetch_count=1)

            channel.basic_consume(
                queue=queue_name,
                on_message_callback=callback,
                auto_ack=auto_ack
            )

            logger.info(f"Started consuming (raw) from queue '{queue_name}'")
            channel.start_consuming()

        except KeyboardInterrupt:
            logger.info(f"Consumer stopped by user")
            channel.stop_consuming()
        except Exception as e:
            logger.exception(f"Error in raw consumer for {queue_name}: {e}")
            raise
