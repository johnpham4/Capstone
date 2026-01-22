import sys
from loguru import logger

from llm_src.infrastructures.messaging.rabbitmq import RabbitMQConsumer, RabbitMQPublisher
from llm_src.domains.processing_request import ProcessingRequest, RequestStatus
from llm_src.domains.events import UserInputReceived, ModelProcessingCompleted, ProcessingFailed
from llm_src.model.inference.inference import LLMInferenceSagemakerEndpoint
from llm_src.settings import settings


QUEUE_NAME = "model_processing_queue"
OUTPUT_QUEUE = "diagram_generation_queue"


class ModelProcessingWorker:
    """Worker to process user input through LLM model"""

    def __init__(self):
        self.publisher = RabbitMQPublisher()
        self.publisher.declare_queue(OUTPUT_QUEUE)

        self.model = LLMInferenceSagemakerEndpoint(
            endpoint_name=settings.SAGEMAKER_ENDPOINT_NAME,
            inference_component_name=getattr(settings, 'SAGEMAKER_INFERENCE_COMPONENT_NAME', None)
        )

    def process_message(self, ch, method, properties, message: dict):
        request_id = message.get('request_id')

        try:
            logger.info(f"Processing model inference for request: {request_id}")

            # Get request and update status to processing
            request = ProcessingRequest.get_by_id(request_id)
            if not request:
                raise ValueError(f"Request {request_id} not found")

            request.update_status(RequestStatus.PROCESSING_MODEL)  # Auto saves

            problem_text = message.get('problem_text', '')

            prompt = self._prepare_prompt(problem_text)

            self.model.set_payload(inputs=prompt)
            response = self.model.inference()

            # Extract model output
            model_output = self._extract_output(response)

            # Parse DSL commands from output
            dsl_commands = self._parse_dsl_commands(model_output)

            # Update database with model results
            request.update_model_output(model_output, dsl_commands)  # Auto saves

            # Publish event to diagram generation queue
            event = ModelProcessingCompleted(
                request_id=request_id,
                model_output=model_output,
                dsl_commands=dsl_commands
            )

            self.publisher.publish(OUTPUT_QUEUE, event.model_dump())

            logger.info(f"Model processing completed for request: {request_id}")

            # Acknowledge message
            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            logger.exception(f"Model processing failed for request {request_id}: {e}")

            # Update status to failed
            request = ProcessingRequest.get_by_id(request_id)
            if request:
                request.update_status(RequestStatus.FAILED, error_message=str(e))  # Auto saves

            # Publish failure event
            failure_event = ProcessingFailed(
                request_id=request_id,
                stage="model",
                error_message=str(e)
            )

            # Acknowledge to remove from queue
            ch.basic_ack(delivery_tag=method.delivery_tag)

    def _prepare_prompt(self, problem_text: str) -> str:
        """Prepare prompt for the model"""
        return """### Instruction:
            Chuyển đổi mô tả hình học tiếng Việt sang GMBL code.

            GMBL Syntax chính:
            - (param (A B C) triangle): Tam giác ABC thường
            - (param (A B C) (iso-tri A)): Tam giác cân tại A
            - (param (A B C) (right-tri B)): Tam giác vuông tại B
            - (define D point (midp A B)): D là trung điểm AB
            - (param D point (on-seg A B)): D nằm trên đoạn AB
            - (param L line (through A)): Đường thẳng qua A
            - (assert (para L1 L2)): L1 song song L2
            - (assert (perp L1 L2)): L1 vuông góc L2
            - (assert (on-line P L)): P nằm trên L
            - (assert (= (uangle A C D) (uangle D C B))): Góc ACD = góc DCB

            Ví dụ:
            Input: "Tam giác ABC, AB = AC"
            Output: (param (A B C) (iso-tri A))

            Input: "Tam giác ABC, điểm D là trung điểm AB, điểm E là trung điểm AC"
            Output: (param (A B C) triangle)
            (define D point (midp A B))
            (define E point (midp A C))

            Bây giờ chuyển đổi:
            {problem_text}

            ### Response:
            """

    def _extract_output(self, response: dict) -> str:
        """Extract text output from model response"""
        if isinstance(response, list) and len(response) > 0:
            return response[0].get('generated_text', '')
        elif isinstance(response, dict):
            return response.get('generated_text', '')
        return str(response)

    def _parse_dsl_commands(self, model_output: str) -> list[str]:
        """Parse DSL commands from model output"""
        # Simple line-based parsing
        lines = [line.strip() for line in model_output.split('\n') if line.strip()]

        # Filter out non-DSL lines (comments, empty lines, etc.)
        dsl_commands = [
            line for line in lines
            if line and not line.startswith('#') and not line.startswith('//')
        ]

        return dsl_commands


def main():
    """Main entry point for model processing worker"""
    logger.info("Starting Model Processing Worker...")

    worker = ModelProcessingWorker()
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
