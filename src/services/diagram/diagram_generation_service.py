import asyncio
import json
from uuid import uuid4

import boto3

from src.config.settings.base import settings
from src.infrastructures.celery.tasks import render_diagram_task


class DiagramService:
    def __init__(self):
        self.sagemaker_client = boto3.client(
            "sagemaker-runtime",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )

    def generate_dsl(
        self,
        user_input: str,
        prompt_template: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        full_prompt = prompt_template.format(query=user_input)
        payload = {
            "messages": [{"role": "user", "content": full_prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "top_k": 50,
        }

        response = self.sagemaker_client.invoke_endpoint(
            EndpointName=settings.SAGEMAKER_ENDPOINT_INFERENCE,
            ContentType="application/json",
            Body=json.dumps(payload),
        )

        result = json.loads(response["Body"].read().decode("utf-8"))

        if "choices" in result and result["choices"]:
            choice = result["choices"][0]
            return choice.get("message", {}).get("content", "") or choice.get("text", "")

        return result.get("generated_text", result.get("text", ""))

    async def render_sync(
        self,
        task_id: str,
        dsl: str,
        epochs: int = 500,
        n_tries: int = 1,
        dpi: int = 150,
        timeout: int = 30,
    ) -> dict:
        return await asyncio.to_thread(
            self.render_blocking,
            task_id,
            dsl,
            epochs,
            n_tries,
            dpi,
            timeout,
        )

    def render_blocking(
        self,
        task_id: str,
        dsl: str,
        epochs: int = 500,
        n_tries: int = 1,
        dpi: int = 150,
        timeout: int = 30,
    ) -> dict:
        task = render_diagram_task.apply_async(
            args=[task_id, dsl],
            kwargs={"epochs": epochs, "n_tries": n_tries, "dpi": dpi},
        )
        return task.get(timeout=timeout)

    def generate_and_render(
        self,
        task_id: str,
        user_input: str,
        prompt_template: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        epochs: int = 500,
        n_tries: int = 1,
        dpi: int = 150,
        timeout: int = 30,
    ) -> dict:
        dsl = self.generate_dsl(
            user_input=user_input,
            prompt_template=prompt_template,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if not dsl.strip():
            return {"error": "LLM returned empty output", "status": "failed"}

        render_result = self.render_blocking(
            task_id=task_id,
            dsl=dsl,
            epochs=epochs,
            n_tries=n_tries,
            dpi=dpi,
            timeout=timeout,
        )

        if isinstance(render_result, dict) and render_result.get("status") == "completed":
            return {
                "dsl": dsl,
                "image": render_result.get("image"),
                "status": "success",
            }

        return {
            "dsl": dsl,
            "error": render_result.get("error", "Rendering failed") if isinstance(render_result, dict) else "Rendering failed",
            "status": "failed",
        }

    async def stream_pipeline_events(
        self,
        user_input: str,
        prompt_template: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        epochs: int = 500,
        n_tries: int = 1,
        dpi: int = 150,
    ):
        request_id = str(uuid4())

        yield {
            "progress": 10,
            "status": "Generating diagram code...",
            "request_id": request_id,
        }

        dsl_output = self.generate_dsl(
            user_input=user_input,
            prompt_template=prompt_template,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        if not dsl_output.strip():
            yield {"progress": 0, "status": "error", "error": "LLM returned empty output"}
            return

        yield {
            "progress": 40,
            "status": "Optimizing geometry...",
            "request_id": request_id,
            "dsl": dsl_output,
        }

        render_result = await self.render_sync(
            task_id=request_id,
            dsl=dsl_output,
            epochs=epochs,
            n_tries=n_tries,
            dpi=dpi,
            timeout=30,
        )

        image_data = render_result.get("image") if isinstance(render_result, dict) else None

        yield {
            "progress": 100,
            "status": "completed",
            "request_id": request_id,
            "user_input": user_input,
            "dsl": dsl_output,
            "image_base64": image_data,
            "svg_content": None,
        }
