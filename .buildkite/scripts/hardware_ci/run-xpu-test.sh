#!/bin/bash

# This script build the XPU docker image and run the offline inference inside the container.
set -ex

omni_source_dir=$(git rev-parse --show-toplevel)

base_image_name="xpu/vllm-omni-ci-base:${VLLM_VERSION:?VLLM_VERSION must be set}"
image_name="xpu/vllm-omni-ci:${BUILDKITE_COMMIT:?BUILDKITE_COMMIT must be set}"
container_name="xpu_${BUILDKITE_COMMIT}_$(
    tr -dc A-Za-z0-9 </dev/urandom | head -c 10
    echo
)"

cd "${omni_source_dir}"
if [ -z "$(docker images -q "${base_image_name}")" ]; then
    docker build --target vllm-base -t "${base_image_name}" --build-arg "VLLM_VERSION=${VLLM_VERSION}" -f docker/Dockerfile.xpu .
fi

# Try building the docker image
docker build --build-arg "VLLM_BASE=${base_image_name}" --build-arg "VLLM_VERSION=${VLLM_VERSION}" -t "${image_name}" -f docker/Dockerfile.xpu .

# Setup cleanup
remove_docker_container() {
    docker rm -f "${container_name}" || true
    docker image rm -f "${image_name}" || true
    docker system prune -f || true
}
trap remove_docker_container EXIT

HF_CACHE="$(realpath ~)/huggingface"
mkdir -p "${HF_CACHE}"
HF_MOUNT="/root/.cache/huggingface"

time timeout -k 30 50m docker run \
    --device /dev/dri:/dev/dri \
    --net=host \
    --ipc=host \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -v "${HF_CACHE}:${HF_MOUNT}" \
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
    pytest -v -s \
        tests/benchmarks/patch/test_patch.py \
        tests/comfyui/test_comfyui_integration.py \
        tests/diffusion/cache/test_cache_backends.py \
        tests/diffusion/distributed/test_cfg_parallel.py \
        tests/diffusion/distributed/test_hsdp.py \
        tests/diffusion/distributed/test_parallel_state_sp_groups.py \
        tests/diffusion/distributed/test_sp_plan_hooks.py \
        tests/diffusion/distributed/test_vae_patch_parallel.py \
        tests/diffusion/lora/test_base_linear.py \
        tests/diffusion/lora/test_lora_manager.py \
        tests/diffusion/models/nextstep_1_1/test_nextstep_cfg_parallel_layout.py \
        tests/diffusion/models/z_image/test_zimage_tp_constraints.py \
        tests/diffusion/test_diffusers_loader.py \
        tests/diffusion/test_diffusion_model_runner.py \
        tests/diffusion/test_diffusion_worker.py \
        tests/diffusion/test_multiproc_executor_concurrency.py \
        tests/diffusion/test_worker_wrapper_base.py \
        tests/distributed/omni_connectors/test_adapter_and_flow.py \
        tests/distributed/omni_connectors/test_chunk_transfer_adapter.py \
        tests/distributed/omni_connectors/test_kv_flow.py \
        tests/distributed/omni_connectors/test_mooncake_transfer_engine_buffer.py \
        tests/distributed/omni_connectors/test_mooncake_transfer_engine_rdma.py \
        tests/distributed/omni_connectors/test_omni_connector_configs.py \
        tests/e2e/offline_inference/test_cache_dit.py \
        tests/e2e/offline_inference/test_t2i_model.py \
        tests/e2e/offline_inference/test_zimage_parallelism.py \
        tests/e2e/online_serving/test_images_generations_lora.py \
        tests/engine/test_async_omni_engine_abort.py \
        tests/entrypoints/openai_api/test_image_server.py \
        tests/entrypoints/openai_api/test_serving_chat_metrics.py \
        tests/entrypoints/openai_api/test_serving_chat_sampling_params.py \
        tests/entrypoints/openai_api/test_serving_speech.py \
        tests/entrypoints/openai_api/test_video_server.py \
        tests/entrypoints/test_async_omni_diffusion_config.py \
        tests/entrypoints/test_cfg_companion_tracker.py \
        tests/entrypoints/test_omni_diffusion.py \
        tests/entrypoints/test_omni_input_preprocessor.py \
        tests/entrypoints/test_omni_llm.py \
        tests/entrypoints/test_omni_new_request_data.py \
        tests/entrypoints/test_omni_stage_diffusion_config.py \
        tests/entrypoints/test_stage_utils.py \
        tests/entrypoints/test_utils.py \
        tests/metrics/test_stats.py \
        tests/model_executor/models/qwen2_5_omni/test_audio_length.py \
        tests/model_executor/models/qwen2_5_omni/test_qwen2_5_omni_embed.py \
        tests/model_executor/models/qwen3_tts/test_cuda_graph_decoder.py \
        tests/test_outputs.py \
        tests/worker/test_gpu_generation_model_runner.py \
        tests/worker/test_omni_gpu_model_runner.py \
        tests/worker/test_process_gpu_memory.py
'
