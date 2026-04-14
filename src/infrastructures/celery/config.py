from celery import Celery
from kombu import Exchange, Queue
from src.config.settings.settings import settings

celery_app = Celery(
    "geouni",
    broker=settings.RABBITMQ_URL,
    backend=settings.REDIS_URL,
    include=["src.infrastructures.celery.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    result_expires=3600,
    task_acks_late=True,
    worker_pool="prefork",
    worker_redirect_stdouts_level="WARNING",
    result_extended=False,

    task_default_queue=settings.DIAGRAM_QUEUE_NAME,
    task_default_exchange=settings.DIAGRAM_QUEUE_EXCHANGE,
    task_default_exchange_type="direct",
    task_default_routing_key=settings.DIAGRAM_QUEUE_ROUTING_KEY,
    task_queues=(
        Queue(
            settings.DIAGRAM_QUEUE_NAME,
            Exchange(settings.DIAGRAM_QUEUE_EXCHANGE, type="direct"),
            routing_key=settings.DIAGRAM_QUEUE_ROUTING_KEY,
        ),
    ),
    task_routes={
        "render_diagram": {
            "queue": settings.DIAGRAM_QUEUE_NAME,
            "routing_key": settings.DIAGRAM_QUEUE_ROUTING_KEY,
        },
    },
)
