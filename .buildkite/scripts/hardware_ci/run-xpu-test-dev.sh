#!/bin/bash

# This script builds the XPU docker image and runs each test command in a
# separate docker container sequentially.  Pass/fail status and logs are
# tracked on the host, and a summary (with failed-test logs) is printed at
# the end.
set -e

image_name="xpu/vllm-omni-ci:test"
container_name_prefix="xpu_test_"

# Try building the docker image
docker build -t "${image_name}" -f docker/Dockerfile.xpu .

# Setup cleanup
remove_docker_container() {
	docker image rm -f "${image_name}" || true
	docker system prune -f || true
}
trap remove_docker_container EXIT

test_commands=(
	# "pytest -v -s tests/benchmarks/test_serve_cli.py" # Must skip, hangs CI
	"pytest -v -s tests/diffusion/attention/test_attention_sp.py"
	"pytest -v -s tests/diffusion/attention/test_flash_attn.py"
	"pytest -v -s tests/diffusion/cache/test_cache_backends.py"
	"pytest -v -s tests/diffusion/distributed/test_cfg_parallel.py"
	"pytest -v -s tests/diffusion/distributed/test_comm.py"
	"pytest -v -s tests/diffusion/distributed/test_parallel_state_sp_groups.py"
	"pytest -v -s tests/diffusion/distributed/test_sp_plan_hooks.py"
	"pytest -v -s tests/diffusion/lora/test_base_linear.py"
	"pytest -v -s tests/diffusion/lora/test_lora_manager.py"
	"pytest -v -s tests/diffusion/models/z_image/test_zimage_tp_constraints.py"
	"pytest -v -s tests/diffusion/test_diffusion_worker.py"
	"pytest -v -s tests/distributed/omni_connectors/test_adapter_and_flow.py"
	"pytest -v -s tests/distributed/omni_connectors/test_basic_connectors.py"
	"pytest -v -s tests/distributed/omni_connectors/test_kv_flow.py"
	"pytest -v -s tests/distributed/omni_connectors/test_omni_connector_configs.py"
	# "pytest -v -s tests/e2e/offline_inference/test_bagel_text2img.py" # Must skip, hangs CI
	"pytest -v -s tests/e2e/offline_inference/test_cache_dit.py"
	"pytest -v -s tests/e2e/offline_inference/test_diffusion_cpu_offload.py"
	"pytest -v -s tests/e2e/offline_inference/test_diffusion_layerwise_offload.py"
	"pytest -v -s tests/e2e/offline_inference/test_diffusion_lora.py"
	"pytest -v -s tests/e2e/offline_inference/test_ovis_image.py"
	"pytest -v -s tests/e2e/offline_inference/test_qwen2_5_omni.py"
	# "pytest -v -s tests/e2e/offline_inference/test_qwen3_omni.py" # Must skip, hangs CI
	"pytest -v -s tests/e2e/offline_inference/test_sequence_parallel.py"
	"pytest -v -s tests/e2e/offline_inference/test_stable_audio_model.py"
	"pytest -v -s tests/e2e/offline_inference/test_t2i_model.py"
	# "pytest -v -s tests/e2e/offline_inference/test_t2v_model.py" # Must skip, hangs CI
	"pytest -v -s tests/e2e/offline_inference/test_teacache.py"
	"pytest -v -s tests/e2e/offline_inference/test_zimage_tensor_parallel.py"
	"pytest -v -s tests/e2e/online_serving/test_async_omni.py"
	"pytest -v -s tests/e2e/online_serving/test_image_gen_edit.py"
	"pytest -v -s tests/e2e/online_serving/test_images_generations_lora.py"
	"pytest -v -s tests/e2e/online_serving/test_qwen3_omni.py"
	"pytest -v -s tests/e2e/online_serving/test_qwen3_omni_expansion.py"
	"pytest -v -s tests/entrypoints/openai_api/test_image_server.py"
	"pytest -v -s tests/entrypoints/openai_api/test_serving_chat_sampling_params.py"
	"pytest -v -s tests/entrypoints/openai_api/test_serving_speech.py"
	"pytest -v -s tests/entrypoints/test_async_omni_diffusion_config.py"
	"pytest -v -s tests/entrypoints/test_omni_diffusion.py"
	"pytest -v -s tests/entrypoints/test_omni_input_preprocessor.py"
	"pytest -v -s tests/entrypoints/test_omni_llm.py"
	"pytest -v -s tests/entrypoints/test_omni_new_request_data.py"
	"pytest -v -s tests/entrypoints/test_omni_stage_diffusion_config.py"
	"pytest -v -s tests/entrypoints/test_stage_utils.py"
	"pytest -v -s tests/model_executor/models/qwen2_5_omni/test_audio_length.py"
	"pytest -v -s tests/test_outputs.py"
	"pytest -v -s tests/worker/test_gpu_generation_model_runner.py"
	"pytest -v -s tests/worker/test_omni_gpu_model_runner.py"
)

# ---------------------------------------------------------------------------
# Utility: derive a log file path from an index and test command
# ---------------------------------------------------------------------------
# Usage: get_log_file <index> <command>
# Extracts the test file path (last argument of the command), replaces / with _,
# strips .py, and returns LOG_DIR/<index>_<sanitized_name>.log
get_log_file() {
	local idx="$1"
	local cmd="$2"
	local test_file="${cmd##* }"
	local sanitized_name="${test_file//\//_}"
	sanitized_name="${sanitized_name%.py}"
	echo "${LOG_DIR}/${idx}_${sanitized_name}.log"
}

# ---------------------------------------------------------------------------
# Host-side log directory
# ---------------------------------------------------------------------------
LOG_DIR="logs/$(date +%s)"
mkdir -p "${LOG_DIR}"

declare -a passed_cmds=()
declare -a failed_cmds=()
declare -a failed_indices=()

total=${#test_commands[@]}

# ---------------------------------------------------------------------------
# Run each test command in its own docker container
# ---------------------------------------------------------------------------
for i in "${!test_commands[@]}"; do
	cmd="${test_commands[$i]}"
	log_file="$(get_log_file "${i}" "${cmd}")"
	container_name="${container_name_prefix}$(tr -dc A-Za-z0-9 </dev/urandom | head -c 10)"

	echo ""
	echo "=========================================="
	echo "[$((i + 1))/${total}] Running: ${cmd}"
	echo "=========================================="

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
		bash -c "
	set -e
	pip install tblib==3.1.0
	cd /workspace/vllm-omni
	${cmd}
" 2>&1 | tee "${log_file}"

	exit_code=${PIPESTATUS[0]}

	if [ "${exit_code}" -eq 0 ]; then
		passed_cmds+=("${cmd}")
		echo ">>> PASSED: ${cmd}"
	else
		failed_cmds+=("${cmd}")
		failed_indices+=("${i}")
		echo ">>> FAILED (exit ${exit_code}): ${cmd}"
	fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "              TEST SUMMARY"
echo "=========================================="
echo "Total:  ${total}"
echo "Passed: ${#passed_cmds[@]}"
echo "Failed: ${#failed_cmds[@]}"

if [ ${#passed_cmds[@]} -gt 0 ]; then
	echo ""
	echo "--- PASSED ---"
	for cmd in "${passed_cmds[@]}"; do
		echo "  [PASS] ${cmd}"
	done
fi

if [ ${#failed_cmds[@]} -gt 0 ]; then
	echo ""
	echo "=========================================="
	echo "           FAILED TESTS"
	echo "=========================================="
	for idx in "${failed_indices[@]}"; do
		cmd="${test_commands[$idx]}"
		log_file="$(get_log_file "${idx}" "${cmd}")"
		echo "  [FAIL] ${cmd}"
		echo "    --> Log file at ${log_file}"
	done

	exit 1
fi
