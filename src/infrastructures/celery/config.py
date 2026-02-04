from celery import Celery
from src.config.settings.base import settings

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
)
