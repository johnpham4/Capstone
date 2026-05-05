# Deploy LLM and Push Image to ECR

Short guide to authenticate AWS CLI, push the vLLM image to ECR, and deploy the SageMaker endpoint.

## Prerequisites

- AWS CLI installed and configured
- Docker installed
- AWS IAM permissions for ECR and SageMaker
- AWS account ID
- 52G memory available

## 1. Configure AWS CLI

```bash
aws login
```

## 2. Get AWS Account ID

```bash
aws sts get-caller-identity
```

## 3. Login to Amazon ECR Public

```bash
aws ecr-public get-login-password --region us-east-1 \
| docker login --username AWS --password-stdin public.ecr.aws
```

## 4. Pull Base Image (example)

```bash
docker pull public.ecr.aws/deep-learning-containers/vllm:0.11.1-gpu-py312-cu129-ubuntu22.04-sagemaker
```

## 5. Login to Amazon ECR Private

Replace <ACCOUNT_ID> with your AWS account ID.

```bash
aws ecr get-login-password --region us-east-1 \
| docker login --username AWS --password-stdin 726101441039.dkr.ecr.us-east-1.amazonaws.com
```

## 6. Create ECR Repository (one-time setup)

```bash
aws ecr create-repository \
  --repository-name vllm \
  --region us-east-1
```

## 7. Tag Docker Image

```bash
docker tag public.ecr.aws/deep-learning-containers/vllm:0.11.1-gpu-py312-cu129-ubuntu22.04-sagemaker \
726101441039.dkr.ecr.us-east-1.amazonaws.com/vllm:0.11.1
```

## 8. Push Image to ECR

```bash
docker push 726101441039.dkr.ecr.us-east-1.amazonaws.com/vllm:0.11.1
```

## 9. Update Account ID in Deploy Script

Replace the account ID inside [src/infrastructures/aws/deploy/huggingface/run.py](src/infrastructures/aws/deploy/huggingface/run.py).

## 10. Deploy Endpoint

```bash
make deploy_endpoint
make del_endpoint
```

## Fix Docker Login Issue (if needed)

Open ~/.docker/config.json and replace:

```json
{
  "credsStore": "desktop.exe"
}
```

With:

```json
{
  "auths": {}
}
```

Retry login after the change.
