import sys
import os
from pathlib import Path
from loguru import logger

from llm_src.infrastructures.messaging.rabbitmq import RabbitMQConsumer
from llm_src.domains.processing_request import ProcessingRequest, RequestStatus
from llm_src.domains.events import ModelProcessingCompleted, DiagramGenerationCompleted, ProcessingFailed
from llm_src.settings import settings

# Import diagram generation components
from llm_src.applications.diagram.parser import Parser
from llm_src.applications.diagram.commands import Command
from llm_src.applications.diagram.optimizer import Optimizer


QUEUE_NAME = "diagram_generation_queue"
OUTPUT_DIR = Path(settings.OUTPUT_DIR) if hasattr(settings, 'OUTPUT_DIR') else Path("./output/diagrams")


class DiagramGenerationWorker:

    def __init__(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.output_dir = OUTPUT_DIR

    def process_message(self, ch, method, properties, message: dict):
        request_id = message.get('request_id')

        try:
            logger.info(f"Generating diagram for request: {request_id}")

            # Get request and update status to generating diagram
            request = ProcessingRequest.get_by_id(request_id)
            if not request:
                raise ValueError(f"Request {request_id} not found")

            request.update_status(RequestStatus.GENERATING_DIAGRAM)  # Auto saves

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
            request.update_diagram_result(str(diagram_path), diagram_points)  # Auto saves

            # Update status to completed
            request.update_status(RequestStatus.COMPLETED)  # Auto saves

            logger.info(f"Diagram generation completed for request: {request_id}")

            # Acknowledge message
            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            logger.exception(f"Diagram generation failed for request {request_id}: {e}")

            # Update status to failed
            request = ProcessingRequest.get_by_id(request_id)
            if request:
                request.update_status(
                    RequestStatus.FAILED,
                    error_message=f"Diagram generation failed: {str(e)}"
                )  # Auto saves

            # Acknowledge to remove from queue
            ch.basic_ack(delivery_tag=method.delivery_tag)


def main():
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
