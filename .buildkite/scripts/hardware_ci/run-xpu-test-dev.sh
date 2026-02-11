#!/bin/bash

# This script builds the XPU docker image and runs each test command in a
# separate docker container sequentially.  Pass/fail status and logs are
# tracked on the host, and a summary (with failed-test logs) is printed at
# the end.
set -e

image_name="xpu/vllm-omni-ci:${BUILDKITE_COMMIT}"
container_name_prefix="xpu_${BUILDKITE_COMMIT}_"

format_duration() {
	local total=$1
	local mins
	local secs
	mins=$(awk "BEGIN {printf \"%d\", ${total} / 60}")
	secs=$(awk "BEGIN {printf \"%.3f\", ${total} - ${mins} * 60}")
	printf '%dm %ss' "${mins}" "${secs}"
}

elapsed_since() {
	local start=$1
	local end
	end=$(date +%s.%N)
	awk "BEGIN {printf \"%.3f\", ${end} - ${start}}"
}

# Try building the docker image
echo "=========================================="
echo "Building docker image: ${image_name}"
echo "=========================================="
build_start=$(date +%s.%N)
docker build -t "${image_name}" -f docker/Dockerfile.xpu .
build_elapsed=$(elapsed_since "${build_start}")
echo ">>> Docker build completed in $(format_duration "${build_elapsed}")"

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
	# "pytest -v -s tests/e2e/online_serving/test_qwen3_omni.py"
	# "pytest -v -s tests/e2e/online_serving/test_qwen3_omni_expansion.py"
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

declare -a statuses=()
declare -a durations=()

total=${#test_commands[@]}
DOCKER_RUN_TIMEOUT=1200

# ---------------------------------------------------------------------------
# Run each test command in its own docker container
# ---------------------------------------------------------------------------
loop_start=$(date +%s.%N)
for i in "${!test_commands[@]}"; do
	cmd="${test_commands[$i]}"
	log_file="$(get_log_file "${i}" "${cmd}")"
	test_start=$(date +%s.%N)
	container_name="${container_name_prefix}$(tr -dc A-Za-z0-9 </dev/urandom | head -c 10)"

	echo ""
	echo "=========================================="
	echo "[$((i + 1))/${total}] Running: ${cmd}"
	echo "=========================================="

	timeout --kill-after=30 "${DOCKER_RUN_TIMEOUT}" \
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

	test_elapsed=$(elapsed_since "${test_start}")
	durations+=("${test_elapsed}")
	formatted_duration="$(format_duration "${test_elapsed}")"

	if [ "${exit_code}" -ne 0 ]; then
		docker rm -f "${container_name}" 2>/dev/null || true
	fi

	if [ "${exit_code}" -eq 0 ]; then
		statuses[$i]="PASS"
		echo ">>> PASSED (${formatted_duration}): ${cmd}"
	elif [ "${exit_code}" -eq 124 ]; then
		statuses[$i]="TIMEOUT"
		echo ">>> TIMED OUT after ${DOCKER_RUN_TIMEOUT}s (${formatted_duration}): ${cmd}"
	elif [ "${exit_code}" -eq 137 ]; then
		statuses[$i]="KILLED"
		echo ">>> KILLED after ${DOCKER_RUN_TIMEOUT}s + 30s grace (${formatted_duration}): ${cmd}"
	else
		statuses[$i]="FAIL"
		echo ">>> FAILED (exit ${exit_code}, ${formatted_duration}): ${cmd}"
	fi
done
total_elapsed=$(elapsed_since "${loop_start}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
passed=0
failed=0
timed_out=0
killed=0
for i in "${!statuses[@]}"; do
	case "${statuses[$i]}" in
		PASS) ((passed++)) || true ;;
		FAIL) ((failed++)) || true ;;
		TIMEOUT) ((timed_out++)) || true ;;
		KILLED) ((killed++)) || true ;;
	esac
done

echo ""
echo "=========================================="
echo "              TEST SUMMARY"
echo "=========================================="
echo "Total:      ${total}"
echo "Passed:     ${passed}"
echo "Failed:     ${failed}"
echo "Timed out:  ${timed_out}"
echo "Killed:     ${killed}"
echo "Docker build time: $(format_duration "${build_elapsed}")"
echo "Total test time:   $(format_duration "${total_elapsed}")"

if [ "${passed}" -gt 0 ]; then
	echo ""
	echo "--- PASSED ---"
	for i in "${!statuses[@]}"; do
		if [ "${statuses[$i]}" = "PASS" ]; then
			echo "  [PASS] ($(format_duration "${durations[$i]}")) ${test_commands[$i]}"
		fi
	done
fi

if [ $((failed + timed_out + killed)) -gt 0 ]; then
	echo ""
	echo "=========================================="
	echo "           FAILED TESTS"
	echo "=========================================="
	for i in "${!statuses[@]}"; do
		if [ "${statuses[$i]}" != "PASS" ]; then
			log_file="$(get_log_file "${i}" "${test_commands[$i]}")"
			echo "  [${statuses[$i]}] ($(format_duration "${durations[$i]}")) ${test_commands[$i]}"
			echo "    --> Log file at ${log_file}"
		fi
	done

	exit 1
fi
