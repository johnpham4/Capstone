## Local vLLM service (OpenAI-compatible)

Repo này gọi LLM theo OpenAI schema tại `LLM_ENDPOINT_URL` (mặc định: `http://localhost:8001/v1`).

Endpoint vLLM cần expose tối thiểu:
- `GET /v1/models`
- `POST /v1/chat/completions`

### 0) Chuẩn bị

1) (Khuyến nghị) Máy có GPU NVIDIA + driver + CUDA runtime phù hợp.
2) Cài Docker Desktop (Windows) hoặc Docker Engine (Linux/WSL2).
3) Tạo file `.env` (tham khảo `.env.example`), tối thiểu:

```env
HF_MODEL_ID=quangne/text2diagram-AceMath-1.5B-Instruct-merged-geometry3k8-8-1-1
HF_TOKEN=... # nếu model private/gated

# Local vLLM (OpenAI-compatible)
LLM_ENDPOINT_URL=http://localhost:8001/v1
```

### 1) Chạy vLLM local

Repo đã có Makefile target:

```bash
make llm_local
```

Chạy background:

```bash
make llm_local_bg
```

Tắt background container:

```bash
docker stop vllm-local
```

Kiểm tra health:

```bash
make llm_local_health
```

Mặc định script sẽ chạy Docker image `vllm/vllm-openai:latest` và map port:
- Host: `8001`
- Container: `8000`

Nếu cần đổi port:

```bash
uv run python scripts/run_vllm_local.py --port 9001
```

### 2) Test gọi OpenAI endpoint

#### 2.1) List models

```bash
curl http://localhost:8001/v1/models
```

#### 2.2) Chat completions

```bash
curl http://localhost:8001/v1/chat/completions \
	-H "Content-Type: application/json" \
	-d '{
		"model": "text2diagram",
		"messages": [{"role": "user", "content": "Hello"}],
		"max_tokens": 64,
		"temperature": 0.0
	}'
```

### 3) Expose ra LAN để máy khác gọi

vLLM trong script đã bind `0.0.0.0` trong container và port mapping sẽ listen trên tất cả interface của máy host.

Bạn cần:

1) Lấy IP LAN của máy chạy vLLM:
	 - Windows: `ipconfig` → IPv4 Address
	 - Linux/WSL: `ip a`

2) Mở firewall inbound port `8001` trên máy chạy vLLM.
	 - Windows Defender Firewall → Inbound Rules → New Rule → Port → TCP 8001 → Allow.

3) Test từ máy khác (cùng mạng):

```bash
curl http://<LAN_IP_CUA_MAY_CHAY_VLLM>:8001/v1/models
```

### 4) Lưu ý khi backend chạy trong Docker

Nếu bạn chạy API/worker bằng `docker compose` thì `LLM_ENDPOINT_URL=http://localhost:8001/v1` sẽ trỏ nhầm vào container.

Một cách đơn giản là đặt trong `.env`:
- Windows/Mac (Docker Desktop): `LLM_ENDPOINT_URL=http://host.docker.internal:8001/v1`
- Hoặc dùng IP LAN của máy host: `LLM_ENDPOINT_URL=http://<LAN_IP>:8001/v1`

