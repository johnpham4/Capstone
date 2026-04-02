
from typing import Dict, Any, Optional, List
from loguru import logger
import httpx


class OpenAIClient:
    """
    Wrapper for OpenAI API calls.

    Supports:
    - Chat completions
    - Async calls
    - Error handling
    """

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=120.0)

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call OpenAI chat completion API.

        Args:
            model: Model name (gpt-4o, gpt-4, o1, etc.)
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Max tokens to generate

        Returns:
            Response dict with 'content' key
        """
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }

            if max_tokens:
                payload["max_tokens"] = max_tokens

            # Add any extra kwargs
            payload.update(kwargs)

            logger.info(f"Calling OpenAI API: model={model}")

            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=payload
            )

            response.raise_for_status()
            result = response.json()

            # Extract content
            content = result["choices"][0]["message"]["content"]

            return {
                "content": content,
                "model": result.get("model"),
                "usage": result.get("usage")
            }

        except httpx.HTTPStatusError as e:
            logger.error(f"OpenAI API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"OpenAI client error: {str(e)}")
            raise

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()

