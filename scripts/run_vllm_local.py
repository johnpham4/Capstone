import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class VllmRunConfig:
    model: str
    host_port: int
    container_port: int
    image: str
    detach: bool
    gpus: str
    trust_remote_code: bool
    max_model_len: int | None
    gpu_memory_utilization: float | None
    dtype: str | None


def _default_model() -> str:
    # Prefer env var, then fall back to app settings if importable.
    env_model = (os.getenv("HF_MODEL_ID") or "").strip()
    if env_model:
        return env_model

    try:
        from src.config.settings import settings  # type: ignore

        return (settings.HF_MODEL_ID or "").strip() or "gpt2"
    except Exception:
        return "gpt2"


def _hf_token() -> str | None:
    token = (os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if token:
        return token

    try:
        from src.config.settings import settings  # type: ignore

        return (settings.HF_TOKEN or "").strip() or None
    except Exception:
        return None


def _build_docker_command(cfg: VllmRunConfig) -> list[str]:
    cmd: list[str] = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        cfg.gpus,
        "-p",
        f"{cfg.host_port}:{cfg.container_port}",
    ]

    token = _hf_token()
    if token:
        cmd += ["-e", f"HUGGING_FACE_HUB_TOKEN={token}"]

    if cfg.detach:
        cmd += ["-d", "--name", "vllm-local"]

    cmd += [
        cfg.image,
        "--model",
        cfg.model,
        "--host",
        "0.0.0.0",
        "--port",
        str(cfg.container_port),
    ]

    if cfg.trust_remote_code:
        cmd += ["--trust-remote-code"]

    if cfg.max_model_len is not None:
        cmd += ["--max-model-len", str(cfg.max_model_len)]

    if cfg.gpu_memory_utilization is not None:
        cmd += ["--gpu-memory-utilization", str(cfg.gpu_memory_utilization)]

    if cfg.dtype:
        cmd += ["--dtype", cfg.dtype]

    return cmd


def _parse_args(argv: list[str]) -> VllmRunConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Run vLLM OpenAI-compatible server locally (Docker). "
            "Default port matches the backend setting: http://localhost:8001/v1"
        )
    )

    parser.add_argument(
        "--model",
        default=_default_model(),
        help="HuggingFace model id (defaults to HF_MODEL_ID env var).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("VLLM_PORT") or "8001"),
        help="Host port to expose the OpenAI API on (default: 8001).",
    )
    parser.add_argument(
        "--container-port",
        type=int,
        default=int(os.getenv("VLLM_CONTAINER_PORT") or "8000"),
        help="Port inside the vLLM container (default: 8000).",
    )
    parser.add_argument(
        "--image",
        default=os.getenv("VLLM_IMAGE") or "vllm/vllm-openai:latest",
        help="Docker image to use (default: vllm/vllm-openai:latest).",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Run container in background (named 'vllm-local').",
    )
    parser.add_argument(
        "--gpus",
        default=os.getenv("VLLM_GPUS") or "all",
        help="Docker --gpus value (default: all).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=(os.getenv("VLLM_TRUST_REMOTE_CODE") or "").lower() in {"1", "true", "yes"},
        help="Pass --trust-remote-code to vLLM.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=(int(os.getenv("VLLM_MAX_MODEL_LEN")) if os.getenv("VLLM_MAX_MODEL_LEN") else None),
        help="Optional: vLLM --max-model-len.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=(
            float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION"))
            if os.getenv("VLLM_GPU_MEMORY_UTILIZATION")
            else None
        ),
        help="Optional: vLLM --gpu-memory-utilization.",
    )
    parser.add_argument(
        "--dtype",
        default=os.getenv("VLLM_DTYPE") or None,
        help="Optional: vLLM --dtype (e.g. float16, bfloat16, auto).",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the docker command and exit.",
    )

    args = parser.parse_args(argv)

    cfg = VllmRunConfig(
        model=str(args.model),
        host_port=int(args.port),
        container_port=int(args.container_port),
        image=str(args.image),
        detach=bool(args.detach),
        gpus=str(args.gpus),
        trust_remote_code=bool(args.trust_remote_code),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=(str(args.dtype).strip() if args.dtype else None),
    )

    if args.print_command:
        docker_cmd = _build_docker_command(cfg)
        print(" ".join(shlex.quote(p) for p in docker_cmd))
        raise SystemExit(0)

    return cfg


def main(argv: list[str]) -> int:
    cfg = _parse_args(argv)

    docker_cmd = _build_docker_command(cfg)

    if not _hf_token():
        # Still ok for public models, but gated models will fail.
        print(
            "[warn] HF token not found (HF_TOKEN/HUGGING_FACE_HUB_TOKEN). "
            "If the model is gated/private, set HF_TOKEN in .env before running.",
            file=sys.stderr,
        )

    print("[info] Starting vLLM container…")
    print("[info] Endpoint:")
    print(f"       http://localhost:{cfg.host_port}/v1")
    print("[info] Health check:")
    print(f"       curl http://localhost:{cfg.host_port}/v1/models")

    try:
        subprocess.run(docker_cmd, check=True)
    except FileNotFoundError:
        print(
            "[error] 'docker' not found. Install Docker Desktop (Windows) or Docker Engine (Linux/WSL).",
            file=sys.stderr,
        )
        return 127
    except subprocess.CalledProcessError as exc:
        return int(exc.returncode or 1)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
