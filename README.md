# 🚀 Push Docker Image to AWS ECR (Public & Private)

This guide explains how to authenticate AWS CLI, pull Docker images, and push them to AWS Elastic Container Registry (ECR).

---

## 📌 Prerequisites

* AWS CLI installed and configured
* Docker installed
* AWS IAM permissions for ECR
* AWS account ID

---

# Requirement:
52G in memory

## 1. Configure AWS CLI

```bash
aws login
```

---

## 2. Get AWS Account ID

```bash
aws sts get-caller-identity
```

---

## 3. Login to Amazon ECR Public

```bash
aws ecr-public get-login-password --region us-east-1 \
| docker login --username AWS --password-stdin public.ecr.aws
```

---

## 4. Pull Base Image (example)

```bash
docker pull public.ecr.aws/deep-learning-containers/vllm:0.11.1-gpu-py312-cu129-ubuntu22.04-sagemaker
```

---

## 5. Login to Amazon ECR Private

Replace `<ACCOUNT_ID>` with your AWS account ID.

```bash
aws ecr get-login-password --region us-east-1 \
| docker login --username AWS --password-stdin 726101441039.dkr.ecr.us-east-1.amazonaws.com
```

---

## 6. Create ECR Repository (one-time setup)

```bash
aws ecr create-repository \
  --repository-name vllm \
  --region us-east-1
```

---

## 7. Tag Docker Image

```bash
docker tag public.ecr.aws/deep-learning-containers/vllm:0.11.1-gpu-py312-cu129-ubuntu22.04-sagemaker \
726101441039.dkr.ecr.us-east-1.amazonaws.com/vllm:0.11.1
```

---

## 8. Push Image to ECR

```bash
docker push 726101441039.dkr.ecr.us-east-1.amazonaws.com/vllm:0.11.1
```

## 9. Replace accountId
replace in file run in folder aws in infrastructure layer
src\infrastructures\aws\deploy\huggingface\run.p

## 10. Deploy
```bash
make deploy_endpoint

make del_endpoint


```
---

## ⚠️ Fix Docker Login Issue (if needed)

If Docker login fails due to credential store:

Open:

```bash
~/.docker/config.json
```

Replace:

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

Then retry login.

---

## 🔁 Restore (Optional)

After successful login, you can restore original Docker config if needed.

---