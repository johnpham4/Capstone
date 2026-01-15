import sys
import os
from pathlib import Path
from loguru import logger

from llm_engineering.infrastructures.db.rabbitmq import RabbitMQConsumer
from llm_engineering.infrastructures.db.processing_request_repository import processing_request_repository
from llm_engineering.domains.events import ModelProcessingCompleted, DiagramGenerationCompleted, ProcessingFailed
from llm_engineering.domains.processing_request import RequestStatus
from llm_engineering.settings import settings

# Import diagram generation components
from llm_engineering.applications.diagram.parser import Parser
from llm_engineering.applications.diagram.commands import Command
from llm_engineering.applications.diagram.optimizer import Optimizer


QUEUE_NAME = "diagram_generation_queue"
OUTPUT_DIR = Path(settings.OUTPUT_DIR) if hasattr(settings, 'OUTPUT_DIR') else Path("./output/diagrams")


class DiagramGenerationWorker:
    """Worker to generate diagrams from DSL commands"""

    def __init__(self):
        # Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.output_dir = OUTPUT_DIR

    def process_message(self, ch, method, properties, message: dict):
        """Process incoming diagram generation request"""
        request_id = message.get('request_id')

        try:
            logger.info(f"Generating diagram for request: {request_id}")

            # Update status to generating diagram
            processing_request_repository.update_status(
                request_id,
                RequestStatus.GENERATING_DIAGRAM
            )

            # Get DSL commands from message
            dsl_commands = message.get('dsl_commands', [])

            if not dsl_commands:
                raise ValueError("No DSL commands provided")

            # Parse and execute DSL commands
            logger.info(f"DSL commands: {dsl_commands}")

            # Step 1: Parse DSL
            parser = Parser()
            parsed_commands = parser.parse_sexprs(dsl_commands)
            logger.info(f"Parsed {len(parsed_commands)} commands")

            # Step 2: Build command structure
            command_reader = Command(dsl_commands)
            points_info = command_reader.points
            logger.info(f"Found {len(points_info)} points")

            # Step 3: Optimize and solve
            optimizer_opts = {
                'epochs': getattr(settings, 'DIAGRAM_OPTIMIZER_EPOCHS', 1000),
                'learning_rate': getattr(settings, 'DIAGRAM_OPTIMIZER_LR', 0.01)
            }

            optimizer = Optimizer(
                command_reader.instructions,
                optimizer_opts,
                verbosity=True
            )

            diagram = optimizer.solve()

            # Step 4: Save diagram
            diagram_filename = f"diagram_{request_id}.png"
            diagram_path = self.output_dir / diagram_filename

            diagram.plot(show=False, save=True, filename=str(diagram_path))
            logger.info(f"Diagram saved to: {diagram_path}")

            # Extract point coordinates
            diagram_points = {
                name: {"x": float(pt.x), "y": float(pt.y)}
                for name, pt in diagram.points.items()
            }

            # Update database with diagram result
            processing_request_repository.update_diagram_result(
                request_id,
                str(diagram_path),
                diagram_points
            )

            # Update status to completed
            processing_request_repository.update_status(
                request_id,
                RequestStatus.COMPLETED
            )

            logger.info(f"Diagram generation completed for request: {request_id}")

            # Acknowledge message
            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            logger.exception(f"Diagram generation failed for request {request_id}: {e}")

            # Update status to failed
            processing_request_repository.update_status(
                request_id,
                RequestStatus.FAILED,
                error_message=f"Diagram generation failed: {str(e)}"
            )

            # Acknowledge to remove from queue
            ch.basic_ack(delivery_tag=method.delivery_tag)


def main():
    """Main entry point for diagram generation worker"""
    logger.info("Starting Diagram Generation Worker...")

    worker = DiagramGenerationWorker()
    consumer = RabbitMQConsumer(
        queue_name=QUEUE_NAME,
        callback=worker.process_message
    )

    try:
        consumer.start_consuming()
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
        consumer.stop_consuming()
    except Exception as e:
        logger.exception(f"Worker error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
