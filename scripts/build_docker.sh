# Build docker image
# Usage: bash scripts/build_docker.sh <image_tag>

echo "Building docker image"
docker build -f dockerfile -t ruhyadi/dspy:$1 .
echo "Image build complete"
