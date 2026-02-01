"""DSL Generator Agent - converts text to Geometry DSL."""

from typing import Dict, Any
import json
import boto3
from loguru import logger

from llm_src.domains.orchestration import Agent, AgentType, AgentState
from llm_src.settings import settings


class DSLGeneratorAgent(Agent):
    """
    Generate Geometry DSL from natural language using fine-tuned LLM.

    Uses SageMaker endpoint with fine-tuned model.
    """

    INSTRUCTION_PROMPT = """Chuyển đổi bài toán hình học tiếng Việt sang Geometry DSL (S-expression syntax).

═══ CÚ PHÁP DSL ═══
[... full prompt từ api/diagram.py ...]

Hãy chuyển đổi đề bài sau:
{query}
"""

    def __init__(self):
        super().__init__(AgentType.DSL_GENERATOR)

        # Initialize SageMaker client
        self.sagemaker_client = boto3.client(
            'sagemaker-runtime',
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
        )

    async def execute(self, state: AgentState) -> AgentState:
        """Generate DSL from user input."""
        state.add_execution_step(self.name)

        try:
            # Build prompt
            full_prompt = self.INSTRUCTION_PROMPT.format(
                query=state.user_input
            )

            # Prepare payload for vLLM endpoint
            payload = {
                "messages": [{"role": "user", "content": full_prompt}],
                "max_tokens": 512,
                "temperature": 0.3,  # Lower temp for more consistent DSL
                "top_p": 0.9,
                "top_k": 50
            }

            logger.info(f"Calling SageMaker endpoint: {settings.SAGEMAKER_ENDPOINT_INFERENCE}")

            # Call endpoint
            response = self.sagemaker_client.invoke_endpoint(
                EndpointName=settings.SAGEMAKER_ENDPOINT_INFERENCE,
                ContentType='application/json',
                Body=json.dumps(payload)
            )

            # Parse response
            result = json.loads(response['Body'].read().decode('utf-8'))

            # Extract DSL
            dsl_output = ""
            if 'choices' in result and len(result['choices']) > 0:
                choice = result['choices'][0]
                if 'message' in choice:
                    dsl_output = choice['message'].get('content', '')
                elif 'text' in choice:
                    dsl_output = choice['text']
            else:
                dsl_output = result.get("generated_text", result.get("text", ""))

            if not dsl_output.strip():
                raise ValueError("LLM returned empty DSL")

            state.dsl = dsl_output.strip()
            logger.info(f"Generated DSL ({len(dsl_output)} chars): {dsl_output[:100]}...")

        except Exception as e:
            error_msg = f"DSL generation failed: {str(e)}"
            state.add_error(error_msg)
            state.dsl_error = error_msg
            logger.error(error_msg)

        return state

    def get_config(self) -> Dict[str, Any]:
        """Get agent configuration."""
        return {
            "type": self.agent_type.value,
            "endpoint": settings.SAGEMAKER_ENDPOINT_INFERENCE,
            "model": settings.HF_MODEL_ID,
            "temperature": 0.3
        }
