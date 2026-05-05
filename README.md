# 🧠 GeoSystem Backend

GeoSystem Backend là hệ thống xử lý bài toán hình học tiếng Viet end-to-end, tu nhan dien ngon ngu tu nhien den sinh bieu dien hinh hoc co cau truc va ve diagram tu dong.

## 🏗️ System Architecture

![System Architecture](./images/architecture.png)

## 🧩 Offline vs Online

- Offline: pipeline tao du lieu (normalize, translate, augment), sinh dataset/DSL, fine-tune LLM, va danh gia.
- Online: serve LLM endpoint, xu ly async tasks, luu metadata/ket qua, va tra ve DSL/diagram.

## 📁 Project Skeleton

```
backend/
  alembic/        # Database migrations
  images/         # Architecture diagrams
  notebooks/      # Experiments and research
  pipeline/       # Offline pipelines (data prep, generation, training)
  scripts/        # Utility scripts
  src/            # API, services, infra, and core app
  Makefile        # Task shortcuts
  pyproject.toml  # Python deps and tooling
  README.md
```

## ⚙️ How to Run

Install dependencies (uv required):

```bash
make install
```

Start services with Docker (postgres, rabbitmq, redis):

```bash
make docker_up
```

Run API endpoint:

```bash
make endpoint
```

LLM local dev (vLLM):

```bash
make llm_local
```

Data pipeline examples:

```bash
make data
make generation
make generate_question
```

Deploy to SageMaker:

```bash
make deploy_endpoint
make del_endpoint
```

Deploy guide: [src/infrastructures/aws/deploy/deploy_llm.md](src/infrastructures/aws/deploy/deploy_llm.md)
