#!/bin/bash

# This script build the CPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

omni_source_dir=$(git rev-parse --show-toplevel)

base_image_name="xpu/vllm-omni-ci-base:${VLLM_VERSION}"
image_name="xpu/vllm-omni-ci:${BUILDKITE_COMMIT}"
container_name="xpu_${BUILDKITE_COMMIT}_$(
    tr -dc A-Za-z0-9 </dev/urandom | head -c 10
    echo
)"

cd "${omni_source_dir}"
[ -z "$(docker images -q ${base_image_name})" ] && docker build --target vllm-base -t "${base_image_name}" --build-arg "VLLM_VERSION=${VLLM_VERSION}" -f docker/Dockerfile.xpu .

# Try building the docker image
docker build --build-arg "VLLM_BASE=${base_image_name}" --build-arg "VLLM_VERSION=${VLLM_VERSION}" -t "${image_name}" -f docker/Dockerfile.xpu .

# Setup cleanup
remove_docker_container() {
    docker rm -f "${container_name}" || true
    docker image rm -f "${image_name}" || true
    docker system prune -f || true
}
trap remove_docker_container EXIT

time timeout -k 30 50m docker run \
    --device /dev/dri:/dev/dri \
    --net=host \
    --ipc=host \
    --privileged \
    -v /dev/dri/by-path:/dev/dri/by-path \
    --entrypoint="" \
    -e HF_TOKEN \
    -e ZE_AFFINITY_MASK \
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
        --ignore=tests/e2e/offline_inference/test_qwen2_5_omni.py \
        --ignore=tests/e2e/offline_inference/test_qwen3_omni.py \
        --ignore=tests/e2e/offline_inference/test_t2v_model.py \
        --ignore=tests/e2e/online_serving/test_qwen3_omni.py \
        --ignore=tests/e2e/online_serving/test_qwen3_omni_expansion.py \
        --ignore=tests/examples/online_serving/test_qwen3_omni.py \
        -k "not test_ring_p2p and not test_sp_correctness"
'
