# Execute docker container
# Usage: bash scripts/exec_docker.sh <image_tag>

echo "Executing docker container"
docker compose up -d\
    && docker exec -it deepstream-py bash\
    &&
echo "Docker container execution complete"