import httpx
from pipeline.settings import settings

def registry_model(model_name: str, prompt: str, version: str, alias: str = ""):
    response = httpx.post(
        f"{settings.BACKEND_URL}/api/v1/registry",
        json={
            "name_hf": model_name,
            "version": version,
            "prompt": prompt,
            "alias": alias,
        },
    )
    response.raise_for_status()
    return response.json()