#!/bin/bash
# deploy.sh — Deploy AutoVision API to AWS EC2
#
# Usage:
#   ./scripts/deploy.sh
#
# Requirements:
#   - AWS CLI configured (aws configure)
#   - Docker running
#   - EC2 instance running with Docker installed

set -e

ECR_REGISTRY="992764024031.dkr.ecr.eu-central-1.amazonaws.com"
ECR_REPO="autovision-api"
REGION="eu-central-1"
IMAGE="$ECR_REGISTRY/$ECR_REPO:latest"

echo "=== AutoVision API — Deploy to AWS ==="
echo "Image: $IMAGE"

# 1. Login to ECR
echo ""
echo "1. Logging in to ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REGISTRY

# 2. Build image
echo ""
echo "2. Building Docker image..."
docker build -t $ECR_REPO .

# 3. Tag image
echo ""
echo "3. Tagging image..."
docker tag $ECR_REPO:latest $IMAGE

# 4. Push to ECR
echo ""
echo "4. Pushing to ECR..."
docker push $IMAGE

echo ""
echo "=== Push complete ==="
echo ""
echo "Next: SSH into EC2 and run:"
echo "  docker stop autovision && docker rm autovision"
echo "  docker pull $IMAGE"
echo "  docker run -d --name autovision -p 8000:8000 $IMAGE"