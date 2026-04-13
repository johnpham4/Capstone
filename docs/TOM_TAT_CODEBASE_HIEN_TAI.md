# Tóm Tắt Codebase Hiện Tại - GeoUni Backend

## 1. Mục Tiêu Dự Án
GeoUni là backend phục vụ bài toán hình học, gồm 2 mảng chính:
- API FastAPI cho xác thực, điều phối luồng LLM, dựng hình từ DSL, lưu lịch sử, chạy tác vụ nền.
- Pipeline dữ liệu và huấn luyện (ZenML) để chuẩn bị dataset, sinh DSL/question và fine-tune mô hình.

## 2. Công Nghệ Chính
- Ngôn ngữ: Python 3.10-3.12
- Web/API: FastAPI, Uvicorn
- ORM/DB: SQLAlchemy async, Alembic, PostgreSQL
- Cache/Queue: Redis, Celery, RabbitMQ, Flower
- LLM/Agent: OpenAI, LangChain, LangGraph
- ML/Pipeline: ZenML, PyTorch, Hugging Face, SageMaker
- UI phụ trợ: Gradio

Tham chiếu: `pyproject.toml`, `src/main.py`, `pipeline/cli.py`.

## 3. Kiến Trúc Tổng Thể
### 3.1 Backend API (`src/`)
- `src/main.py`: khởi tạo app FastAPI, CORS, lifespan, init DB khi startup, đóng Redis khi shutdown.
- `src/api/endpoints.py`: đăng ký toàn bộ router.
- `src/api/routes/`: nhóm endpoint theo domain.
- `src/services/`: logic nghiệp vụ.
- `src/repositories/`: truy cập dữ liệu theo repository pattern.
- `src/models/`: DTO và ORM.
- `src/infrastructures/`: kết nối DB, Redis, Celery, OpenAI, AWS.
- `src/config/settings/base.py`: cấu hình tập trung từ `.env`.

### 3.2 Pipeline ML (`pipeline/`)
- `pipeline/cli.py`: cổng chạy pipeline bằng cờ CLI.
- `pipeline/pipelines/`: định nghĩa các ZenML pipeline.
- `pipeline/steps/`: các step cụ thể (data prep, dataset, upload, training).
- `pipeline/services/`: dịch vụ sinh dataset, dịch thuật, finetune.
- `pipeline/domain/`: mô hình dữ liệu pipeline.

### 3.3 Dữ liệu/Migration/Tiện ích
- `alembic/`: migration DB.
- `dataset/`: dữ liệu nguồn, train/test, review.
- `scripts/seed_users.py`: seed user demo.
- `notebooks/`: notebook thử nghiệm/inference.

## 4. API Hiện Có
### 4.1 Auth (`src/api/routes/auth.py`)
- `POST /api/v1/auth/token`: login OAuth2 form.
- `POST /api/v1/auth/login`: login JSON cho frontend.
- `POST /api/v1/auth/logout`: thu hồi token hiện tại.
- `GET /api/v1/auth/me`: lấy thông tin user hiện tại.
- `POST /api/v1/auth/register`: đăng ký tài khoản.

### 4.2 Diagram (`src/api/routes/diagram.py`)
- `GET /api/v1/diagrams/stream-pipeline`: stream SSE quá trình sinh DSL + render ảnh.
- Có rate limit (`rate_limit_diagram`) và lưu history vào DB.

### 4.3 Orchestration (`src/api/routes/orchestration.py`)
- `POST /api/v1/orchestration`: chạy đồng bộ pipeline điều phối (rewrite -> diagram -> solve).
- `GET /api/v1/orchestration/stream`: stream SSE theo từng stage (`rewrite`, `diagram`, `solver`, `done`, `error`).

### 4.4 History (`src/api/routes/history.py`)
- `GET /api/v1/history`: phân trang lịch sử request của user.
- `GET /api/v1/history/{request_id}`: chi tiết (request + diagram + solution).
- `DELETE /api/v1/history/{request_id}`: xóa lịch sử thuộc user.

### 4.5 Tasks (`src/api/routes/tasks.py`)
- `POST /api/v1/tasks/diagrams/render`: đẩy tác vụ render vào Celery.
- `GET /api/v1/tasks/status/{celery_task_id}`: kiểm tra trạng thái task.
- `GET /api/v1/tasks/workers/status`: trạng thái worker.
- `GET /api/v1/tasks/tasks/active`: danh sách task đang chạy.

## 5. Data Layer
### 5.1 ORM (`src/models/orm.py`)
- `UserModel`: người dùng.
- `RequestModel`: mỗi request người dùng (mode, status, latency).
- `DiagramModel`: DSL + ảnh render cho 1 request.
- `SolutionModel`: lời giải cho 1 request.
- `RegistryModel`: thông tin model version/prompt.

### 5.2 Repository (`src/repositories/`)
- `base.py`: CRUD chung (`create`, `get_by_id`, `get_all`, `update`, `delete`, `count`).
- `user.py`, `request.py`, `diagram.py`, `solution.py`, `registry.py`: truy vấn đặc thù từng bảng.

## 6. Service Layer
### 6.1 Điều phối tổng (`src/services/orchestration/orchestrator.py`)
- Dùng LangGraph để chạy state machine:
  - `parse` (RewriterAgent)
  - `diagram` (DiagramAgent)
  - `solve` (SolverAgent)
- Hỗ trợ chạy đồng bộ (`execute`) và stream (`stream_execute`).

### 6.2 Agent
- `rewriter_agent.py`: chuẩn hóa đề và xác định mode.
- `diagram_agent.py`: gọi dịch vụ sinh DSL + render.
- `solver_agent.py`: sinh lời giải (sync/stream).
- `ocr_agent.py`: khung OCR (chưa thấy dùng rõ trong route hiện tại).

### 6.3 Diagram Engine (`src/services/diagram/`)
- `dsl_parser.py`: parse S-expression DSL.
- `diagram_builder.py`: đổi DSL sang instruction nội bộ.
- `optimizer.py`: tối ưu ràng buộc hình học bằng PyTorch.
- `initializer.py`: khởi tạo điểm ban đầu (giảm rơi vào nghiệm xấu).
- `matplotlib_renderer.py`: vẽ ảnh từ nghiệm tối ưu.
- `model/`: định nghĩa entities/types/instructions/value objects.

### 6.4 Auth/History/Task
- `src/services/auth/auth.py`: login/register/logout, JWT.
- `src/services/history/service.py`: lưu tiến trình và truy xuất lịch sử.
- `src/services/tasks/queue.py`: giao tiếp Celery cho task render.
- `src/services/registry/registry.py`: thao tác model registry.

## 7. Hạ Tầng và Cấu Hình
### 7.1 Database
- `src/infrastructures/database/session.py`: async engine/session, `get_db`, `init_db`.

### 7.2 Redis
- `src/infrastructures/redis/connection.py`: singleton async Redis connector.
- `src/infrastructures/redis/cache.py`: blacklist token khi logout.

### 7.3 Celery
- `src/infrastructures/celery/config.py`: cấu hình broker/backend/worker.
- `src/infrastructures/celery/tasks.py`: worker task render diagram.

### 7.4 LLM/AWS
- `src/infrastructures/llm/openai_client.py`: wrapper gọi OpenAI.
- `src/infrastructures/aws/`: role/deploy endpoint SageMaker.

### 7.5 Settings
- `src/config/settings/base.py`: tập trung biến môi trường cho DB, JWT, OpenAI, Redis, RabbitMQ, AWS, HF, CORS.

## 8. Pipeline Dữ Liệu và Huấn Luyện
### 8.1 CLI (`pipeline/cli.py`)
Các cờ chính:
- `--run-prepare-data`
- `--run-upload-dataset`
- `--run-generate-gmbl`
- `--run-generate-questions`
- `--run-finetune`

### 8.2 Pipelines (`pipeline/pipelines/`)
- `data_preparation.py`: tải dữ liệu SynthGeo, dịch caption, lưu JSON, lọc tam giác.
- `dataset_generation.py`: nạp nguồn, tạo prompt, sinh bộ train/test DSL.
- `question_generation.py`: sinh bộ câu hỏi từ dữ liệu hiện có.
- `dataset_upload.py`: upload dataset lên Hugging Face.
- `training.py`: orchestration step huấn luyện.
- `evaluating.py`: khung đánh giá.

### 8.3 Steps/Services
- `pipeline/steps/data_prep/`: download, translate, filter, save.
- `pipeline/steps/dataset/`: load data, generate dataset/question, save.
- `pipeline/steps/upload/`: upload HF.
- `pipeline/steps/training/`: train + registry.
- `pipeline/services/datasets/`: extraction, generation, parser output, rewrite câu hỏi.
- `pipeline/services/preprocessing/translation.py`: translator domain hình học.
- `pipeline/services/peft_finetuning/`, `pipeline/services/unsloth_finetune/`: train/inference/SageMaker.

## 9. Migration DB (Alembic)
Trong `alembic/versions/`:
- `23bad07600c8_initial_tables.py`: tạo bảng ban đầu.
- `754ca6684f14_initial_tables.py`: bỏ cột `cache_hit`.
- `75ac6d7e33e6_initial_tables.py`: bỏ một số cột metadata ở `diagrams/solutions`.
- `d90defe07520_upgrade_tables.py`: bỏ cột `users.full_name`.
- `b3e1c7b2a9f1_add_model_versions_registry.py`: file hiện đang rỗng.

`alembic/env.py` chuyển URL async sang sync cho Alembic (`+asyncpg` -> mặc định sync driver).

## 10. Vận Hành/DevOps
- `Dockerfile`: multi-stage, target `api`, `worker`, `flower`.
- `Dockerfile.dev`: môi trường dev nhẹ.
- `compose.yaml`: hiện phần infra chính (postgres/rabbitmq/redis/api/worker/flower) đang comment; còn active `mysql` + `zenml-server`.
- `Makefile`: nhóm lệnh cài đặt, chạy pipeline, celery, docker, migration, deploy AWS.

## 11. Script/Notebook/Tiện Ích
- `scripts/seed_users.py`: init DB và seed user demo `johndoe`.
- `ui.py`: Gradio tool để review dữ liệu lỗi/ảnh.
- `split_full_json.py`, `test_diagram.py`: script tiện ích cho dữ liệu/diagram.
- `notebooks/acemath.ipynb`, `notebooks/llm_inference.ipynb`: thử nghiệm dữ liệu và inference.

## 12. Điểm Cần Lưu Ý Khi Bảo Trì
- `src/api/routes/diagram.py` và `src/services/orchestration/agents/diagram_agent.py` import `src.services.diagram.generation.DiagramService`, nhưng trong thư mục `src/services/diagram/` hiện chưa có file `generation.py`.
- File migration `alembic/versions/b3e1c7b2a9f1_add_model_versions_registry.py` đang rỗng, nên cần kiểm tra lại lịch sử migration.
- `compose.yaml` đang thiên về stack ZenML (MySQL + ZenML server), trong khi stack backend chính bị comment.
- Rate limiter phụ thuộc Redis; khi Redis down thì cơ chế giới hạn sẽ degrade gracefully (không chặn request).

## 13. Tóm Tắt Nhanh Luồng Nghiệp Vụ
### 13.1 Luồng Orchestration
1. Client gửi đề bài.
2. Rewriter chuẩn hóa đề và mode.
3. Diagram agent sinh DSL và ảnh (nếu mode cần diagram).
4. Solver sinh lời giải (nếu mode `both`).
5. Kết quả và metadata được lưu vào history.

### 13.2 Luồng Task Render Bất Đồng Bộ
1. Client gọi endpoint queue task.
2. Celery worker xử lý render.
3. Client poll endpoint status để lấy tiến trình/kết quả.

---
Tài liệu này phản ánh code hiện tại trong workspace tại thời điểm tổng hợp (08-04-2026).
