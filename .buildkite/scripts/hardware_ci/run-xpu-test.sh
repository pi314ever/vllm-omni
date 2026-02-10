#!/bin/bash

# This script build the CPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

image_name="xpu/vllm-omni-ci:${BUILDKITE_COMMIT}"
container_name="xpu_${BUILDKITE_COMMIT}_$(
    tr -dc A-Za-z0-9 </dev/urandom | head -c 10
    echo
)"

# Try building the docker image
docker build -t ${image_name} -f docker/Dockerfile.xpu .

# Setup cleanup
remove_docker_container() {
    docker image rm -f "${image_name}" || true
    docker system prune -f || true
}
trap remove_docker_container EXIT

# Run the image and test offline inference/tensor parallel
docker run \
    --device /dev/dri:/dev/dri \
    --net=host \
    --ipc=host \
    --privileged \
    --rm \
    -v /dev/dri/by-path:/dev/dri/by-path \
    --entrypoint="" \
    -e "HF_TOKEN=${HF_TOKEN}" \
    -e "ZE_AFFINITY_MASK=${ZE_AFFINITY_MASK}" \
    --name "${container_name}" \
    "${image_name}" \
    bash -c '
    set -e
    echo $ZE_AFFINITY_MASK
    pip install tblib==3.1.0
    cd /workspace/vllm-omni
    pytest -v -s tests/ \
        --ignore=tests/benchmarks/test_serve_cli.py \
        --ignore=tests/e2e/offline_inference/test_bagel_text2img.py \
        --ignore=tests/e2e/offline_inference/test_qwen3_omni.py \
        --ignore=tests/e2e/offline_inference/test_t2v_model.py
'
