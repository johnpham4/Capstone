# GeoUni - Geometry Problem-Solving LLM

An end-to-end LLM application for automated geometry problem solving, combining fine-tuned language models with constraint-based diagram generation.

## Overview

GeoUni fine-tunes Qwen2.5-7B on geometric problem datasets to understand and solve geometry problems. The system integrates:

- **Fine-tuned LLM**: LoRA/QLoRA fine-tuning on geometry datasets achieving training loss of 0.021
- **Diagram Builder**: Constraint-based optimization for generating geometric figures from natural language
- **LLMOps Pipeline**: ZenML orchestration with Comet ML tracking for reproducible ML workflows
- **Cloud Deployment**: AWS SageMaker inference endpoints with TGI (Text Generation Inference)

**Model**: [minn4/GeoUni-Qwen2.5-7B](https://huggingface.co/minn4/GeoUni-Qwen2.5-7B)

## Project Structure

```
GeoUni/
├── llm_engineering/          # Core LLM application
│   ├── applications/         # Business logic (datasets, preprocessing, networks)
│   ├── domains/             # Domain models (DDD pattern)
│   ├── infrastructures/     # External services (AWS, DB)
│   └── model/               # Model training, inference, evaluation
├── pipelines/               # ZenML ML pipelines
│   ├── data_preparation.py  # Data cleaning and translation
│   ├── dataset_generation.py # Dataset creation
│   ├── training.py          # Model fine-tuning
│   └── evaluating.py        # Model evaluation
├── steps/                   # ZenML pipeline steps
├── geo-model-builder/       # Geometry diagram builder
│   ├── src/                 # Builder implementation
│   └── problems/            # IMO and test problems
├── notebooks/               # Jupyter notebooks for experiments
├── configs/                 # Pipeline configurations
└── tools/                   # CLI utilities
```

## Features

### 1. LLM Fine-Tuning
- Fine-tune Qwen2.5-7B using Unsloth with LoRA adapters
- Dataset preparation from SynthGeo and custom geometry problems
- Experiment tracking with Comet ML
- Support for QLoRA (4-bit quantization) for efficient training

### 2. Geometry Diagram Generation
- Parse natural language descriptions into geometric constraints
- Constraint-based optimization using TensorFlow/NumPy
- Support for IMO (International Mathematical Olympiad) problems
- Interactive diagram builder CLI and server

### 3. LLMOps Pipeline
- **Data Preparation**: Download, filter, and translate geometry datasets
- **Dataset Generation**: Create instruction-tuning datasets
- **Training**: Fine-tune models with configurable hyperparameters
- **Evaluation**: Assess model performance on test sets
- **Deployment**: Deploy to AWS SageMaker endpoints

### 4. Cloud Deployment
- AWS SageMaker inference endpoints with TGI
- Auto-scaling policies for cost optimization
- FastAPI integration for inference APIs
- Docker containerization support

## Quick Start

### Prerequisites
- Python 3.11+
- AWS Account (for deployment)
- HuggingFace Account (for model access)

### Installation

```bash
# Install dependencies
pip install uv
uv pip install -e .

# For AWS deployment
uv pip install -e ".[aws]"
```

### Configuration

Create `.env` file with required credentials:

```bash
# HuggingFace
HF_TOKEN=your_huggingface_token
HF_MODEL_ID=minn4/GeoUni-Qwen2.5-7B

# AWS
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_ARN_ROLE=your_sagemaker_role_arn

# Comet ML
COMET_API_KEY=your_comet_api_key
COMET_PROJECT=geouni
```

### Run Pipelines

```bash
# Data preparation
python tools/run.py --pipeline data_preparation

# Generate dataset
python tools/run.py --pipeline dataset_generation

# Train model
python tools/run.py --pipeline training

# Evaluate model
python tools/run.py --pipeline evaluating
```

### Deploy to AWS SageMaker

```bash
# Deploy endpoint
make deploy_endpoint

# Check endpoint status
python -c "from llm_engineering.model.utils import ResourceManager; rm = ResourceManager(); print(rm.endpoint_exists('text2diagram-llm-endpoint'))"
```

### Run Inference

```python
from llm_engineering.model.inference import LLMInferenceSagemakerEndpoint

inference = LLMInferenceSagemakerEndpoint(
    endpoint_name="text2diagram-llm-endpoint"
)

response = inference.inference(
    prompt="Given triangle ABC with AB=5, BC=6, AC=7. Find the area.",
    max_new_tokens=512
)
print(response)
```

## Architecture

The project follows **Domain-Driven Design (DDD)** with **Clean Architecture** principles:

- **Domains**: Core business logic and entities
- **Applications**: Use cases and application services
- **Infrastructures**: External integrations (AWS, databases)
- **Interfaces**: API endpoints and user interfaces

## Technologies

- **ML/DL**: PyTorch, Transformers, Unsloth, LoRA
- **MLOps**: ZenML, Comet ML, MLflow
- **Cloud**: AWS SageMaker, Amazon S3
- **Backend**: FastAPI, boto3
- **Tools**: Docker, Make, Poetry/uv

## License

See individual component licenses:
- Main project: See LICENSE
- geo-model-builder: See geo-model-builder/LICENSE
- AWS CLI: See aws/THIRD_PARTY_LICENSES

## Acknowledgments

- SynthGeo dataset for training data
- Unsloth for efficient fine-tuning
- AWS SageMaker for model deployment
- HuggingFace for model hosting and TGI