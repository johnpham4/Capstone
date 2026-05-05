```markdown
# 🧠 GeoSystem Backend

**GeoSystem Backend** là một hệ thống xử lý bài toán hình học tiếng Việt end-to-end, từ việc phân tích ngôn ngữ tự nhiên (NLP) đến sinh biểu diễn hình học có cấu trúc và trực quan hóa chính xác.

Hệ thống kết hợp giữa **LLM (Large Language Models)**, **symbolic geometry** và **data pipelines**, cho phép chuyển đổi đề bài hình học thành dạng máy hiểu được và dựng lại diagram một cách tự động.

---

## 🚀 Key Idea

Bài toán hình học thường được viết dưới dạng ngôn ngữ tự nhiên (tiếng Việt), nhưng để máy hiểu và xử lý thì cần chuyển sang dạng cấu trúc.

👉 GeoSystem giải quyết bài toán này theo pipeline:

1. **Vietnamese NLP Parsing**
   Phân tích đề bài để hiểu ngữ nghĩa và cấu trúc câu.

2. **Entity & Relation Extraction**
   Trích xuất các thành phần hình học:
   - Điểm (A, B, C, …)
   - Đường thẳng, đoạn thẳng
   - Đường tròn
   - Quan hệ (vuông góc, song song, tiếp xúc, …)

3. **Geometry DSL Generation (S-expression)**
   Chuyển đổi sang dạng biểu diễn hình học có cấu trúc để máy xử lý.

4. **Symbolic Geometry Rendering**
   Sinh diagram chính xác từ DSL.

---

## 🏗️ System Architecture

Hệ thống gồm 2 phần chính:

### 1. Offline Pipeline
- Chuẩn bị và làm sạch dữ liệu
- Sinh dữ liệu hình học (data generation)
- Fine-tune LLM cho bài toán parsing

### 2. Online Service
- Serve LLM endpoints
- Xử lý async tasks (queue-based workers)
- Trả về kết quả DSL hoặc diagram

---

## 📁 Project Structure

```

backend/
alembic/        # Database migrations
images/         # Architecture diagrams & assets
notebooks/      # Experiments & research
pipeline/       # Data processing & generation pipelines
scripts/        # Utility scripts
src/            # Core application code
Makefile
pyproject.toml
README.md

````

---

## ⚙️ Getting Started

### Install dependencies
```bash
make install
````

### Start services with Docker

```bash
make docker_up
```

### Run API endpoint

```bash
make endpoint
```

---

## 🤖 LLM Development

Chạy LLM local để development:

```bash
make llm_local
```

---

## 📊 Data Pipeline

Ví dụ các pipeline:

```bash
make data
make generation
make generate_question
```

---

## ☁️ Deployment (AWS SageMaker)

Chi tiết tại:

```
src/infrastructures/aws/deploy/deploy_llm.md
```

Commands:

```bash
make deploy_endpoint
make del_endpoint
```

---

## ✨ Highlights

* Xử lý **đề bài hình học tiếng Việt**
* Chuyển đổi sang **Geometry DSL (S-expression)**
* Kết hợp **LLM + symbolic reasoning**
* Kiến trúc **modular: pipeline + serving**
* Hỗ trợ **async processing + scalable deployment**

---

## 📌 Future Work

* Improve parsing accuracy với structured prompting / fine-tuning
* Hỗ trợ nhiều dạng bài hình học hơn
* Tích hợp solver để giải bài toán, không chỉ vẽ hình
* Scaling hệ thống (distributed workers, caching, batching)

---

```
```
