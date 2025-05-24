#!/bin/bash

# CI-Examples/run_benchmarks.sh
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2023 Intel Corporation
#
# This script orchestrates benchmarking for GPU applications in different modes:
# 1. Native Linux
# 2. Gramine Direct (unprotected)
# 3. Gramine SGX (shared enclave model)
#
# It also serves as a guide, providing commands and suggestions for profiling.

set -u # Treat unset variables as an error
# set -e # Exit on any error - might be too strict for a multi-stage script

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
GRAMINE_ROOT_DIR="${SCRIPT_DIR}/../" # Adjust if CI-Examples is not at root
RESULTS_DIR="${SCRIPT_DIR}/benchmark_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Application specific details
SHARED_ENCLAVE_DIR="${SCRIPT_DIR}/shared-enclave"
CLIENT_ENCLAVE_DIR="${SCRIPT_DIR}/client-enclave"
NATIVE_APPS_DIR="${SCRIPT_DIR}/native-apps" # New directory for native benchmark sources/binaries

# SGX Signing (User MUST configure this)
SIGNER_KEY_FILE="${SCRIPT_DIR}/YOUR_SIGNER_KEY.pem" # REPLACE THIS

# Benchmark parameters
NUM_RUNS=5 # Number of times to run each benchmark for averaging (application should loop internally for more stable timing)
VECTOR_SIZE_ELEMENTS=1048576 # Example size for vector add (e.g., 2^20 floats)
ONNX_MODEL_NAME="mobilenetv2-7.onnx" # Used for logging, actual model path in manifest/service
GEMM_M=512
GEMM_N=512
GEMM_K=512

# --- Helper Functions ---
log_message() {
    echo ""
    echo "--------------------------------------------------------------------------"
    echo "$(date +"%Y-%m-%d %H:%M:%S") - $1"
    echo "--------------------------------------------------------------------------"
}

# Function to create necessary directories
setup_dirs() {
    mkdir -p "${RESULTS_DIR}"
    mkdir -p "${NATIVE_APPS_DIR}/bin"
    # Ensure client/server build directories exist for logs, if not built by this script
    mkdir -p "${CLIENT_ENCLAVE_DIR}/logs"
    mkdir -p "${SHARED_ENCLAVE_DIR}/logs"
}

# Function to check for required executables
check_executable() {
    local exec_path="$1"
    local exec_name="$2"
    if [ ! -x "$exec_path" ]; then
        log_message "ERROR: $exec_name executable not found or not executable at $exec_path."
        echo "Please build all applications first (e.g., using 'make' in CI-Examples and native-apps)."
        exit 1
    fi
}

# Stubs for run functions - to be implemented
run_native_vector_add() {
    log_message "Running Native Vector Add (Size: $1)"
    local size="$1"
    local mode="native"
    local app="vector_add"
    local result_prefix="${RESULTS_DIR}/${app}_${mode}_size${size}_${TIMESTAMP}"
    local size="$1"
    local mode="native"
    local app="vector_add"
    local result_prefix="${RESULTS_DIR}/${app}_${mode}_size${size}_${TIMESTAMP}"
    local exec_path="${NATIVE_APPS_DIR}/bin/vector_add_native_benchmark" # Updated name
    
    check_executable "$exec_path" "Native Vector Add Benchmark"
    
    log_message "Executing Native Vector Add (Size: ${size}). Log: ${result_prefix}.log, Time: ${result_prefix}.time"
    # Application itself takes size as argument.
    # It should loop internally NUM_RUNS if we want to average internal timings.
    # For /usr/bin/time, it measures a single invocation.
    # If the app loops N times, /usr/bin/time gives total for N runs.
    # The app's internal cudaEvent_t timing will be for one GPU operation.
    /usr/bin/time -v -o "${result_prefix}.time" "$exec_path" "$size" > "${result_prefix}.log" 2>&1
    cmd_exit_code=$?

    if [ $cmd_exit_code -ne 0 ]; then
        log_message "ERROR: Native Vector Add failed with exit code ${cmd_exit_code}."
        echo "See log: ${result_prefix}.log"
        return 1
    fi
    log_message "Native Vector Add finished. Results in ${result_prefix}.*"
    echo "Profiling hint: perf record -e cycles,instructions -g -- \"$exec_path\" \"$size\""
    echo "Profiling hint (GPU): sudo /usr/local/cuda/bin/nvprof --log-file \"${result_prefix}_nvprof.log\" \"$exec_path\" \"$size\""
    echo "Profiling hint (GPU with Nsight Systems): sudo /opt/nvidia/nsight-systems/latest/bin/nsys profile -o \"${result_prefix}_nsys_report\" \"$exec_path\" \"$size\""
    return 0
}

run_gramine_direct_vector_add() {
    log_message "Running Gramine-Direct Vector Add (Size: $1)"
    local size="$1"
    local mode="gramine_direct"
    local app="vector_add"
    local result_prefix="${RESULTS_DIR}/${app}_${mode}_size${size}_${TIMESTAMP}"
    local exec_path_in_gramine="/benchmark_apps/vector_add_native_benchmark" # Path inside Gramine manifest
    local manifest_path="${NATIVE_APPS_DIR}/vector_add_native.direct.manifest.template" # Using template directly for direct mode
    
    # Ensure the native executable (that gramine-direct will run) exists
    check_executable "${NATIVE_APPS_DIR}/bin/vector_add_native_benchmark" "Native Vector Add Benchmark (for Gramine-Direct)"
    if [ ! -f "$manifest_path" ]; then
        log_message "ERROR: Direct manifest not found at $manifest_path"
        return 1
    fi

    log_message "Executing Gramine-Direct Vector Add (Size: ${size}). Log: ${result_prefix}.log, Time: ${result_prefix}.time"
    # The executable path for gramine-direct is taken from the manifest's libos.entrypoint
    # Arguments to the entrypoint are passed after 'gramine-direct manifest_file'
    /usr/bin/time -v -o "${result_prefix}.time" \
        gramine-direct "$manifest_path" "$size" > "${result_prefix}.log" 2>&1
    cmd_exit_code=$?

    if [ $cmd_exit_code -ne 0 ]; then
        log_message "ERROR: Gramine-Direct Vector Add failed with exit code ${cmd_exit_code}."
        echo "See log: ${result_prefix}.log"
        return 1
    fi
    log_message "Gramine-Direct Vector Add finished. Results in ${result_prefix}.*"
    echo "Profiling hint (Direct): perf record -e cycles,instructions -g -- gramine-direct \"$manifest_path\" \"$size\""
    echo "Profiling hint (GPU, Direct): sudo /usr/local/cuda/bin/nvprof --log-file \"${result_prefix}_nvprof.log\" gramine-direct \"$manifest_path\" \"$size\""
    return 0
}

run_sgx_shared_vector_add() {
    local masking_mode="$1" # "aes_gcm" or "none"
    local size="$2"
    log_message "Running Gramine SGX Shared Vector Add (Size: ${size}, Masking: ${masking_mode})"
    
    local app_name="vector_add"
    local client_app_exec="client_app" # Executable name
    local client_manifest="client-enclave.manifest.sgx" # Base manifest name
    local result_prefix="${RESULTS_DIR}/${app_name}_sgx_shared_masking_${masking_mode}_size${size}_${TIMESTAMP}"
    # Server log can be shared for now, or made unique if server behavior changes with client masking mode
    local server_log="${SHARED_ENCLAVE_DIR}/logs/${app_name}_server_sgx_${TIMESTAMP}.log" 
    local client_log="${CLIENT_ENCLAVE_DIR}/logs/${app_name}_client_sgx_masking_${masking_mode}_size${size}_${TIMESTAMP}.log"
    local server_pid_file="${SHARED_ENCLAVE_DIR}/logs/server_${app_name}_sgx.pid"

    mkdir -p "${SHARED_ENCLAVE_DIR}/logs" "${CLIENT_ENCLAVE_DIR}/logs"

    # Server manifest signing (common for all masking modes of this app type)
    # Client manifest signing is also common, client app itself interprets --masking
    if [ ! -f "${SHARED_ENCLAVE_DIR}/shared-enclave.manifest.sgx" ] || \
       [ ! -f "${CLIENT_ENCLAVE_DIR}/${client_manifest}" ]; then
        log_message "Signing SGX manifests for ${app_name}..."
        gramine-sgx-sign \
            --manifest "${SHARED_ENCLAVE_DIR}/shared-enclave.manifest.template" \
            --output "${SHARED_ENCLAVE_DIR}/shared-enclave.manifest.sgx" \
            --key "${SIGNER_KEY_FILE}"
        gramine-sgx-sign \
            --manifest "${CLIENT_ENCLAVE_DIR}/client-enclave.manifest.template" \
            --output "${CLIENT_ENCLAVE_DIR}/${client_manifest}" \
            --key "${SIGNER_KEY_FILE}"
        echo "Manifests signed."
    else
        echo "Manifests for ${app_name} already signed, skipping."
    fi
    gramine-sgx-sign \
        --manifest "${SHARED_ENCLAVE_DIR}/shared-enclave.manifest.template" \
        --output "${SHARED_ENCLAVE_DIR}/shared-enclave.manifest.sgx" \
        --key "${SIGNER_KEY_FILE}"
    gramine-sgx-sign \
        --manifest "${CLIENT_ENCLAVE_DIR}/client-enclave.manifest.template" \
        --output "${CLIENT_ENCLAVE_DIR}/client-enclave.manifest.sgx" \
        --key "${SIGNER_KEY_FILE}"
    echo "Manifests signed."

    log_message "Starting Shared Enclave Service for ${app_name}..."
    (cd "${SHARED_ENCLAVE_DIR}" && \
        gramine-sgx shared-enclave.manifest.sgx > "${server_log}" 2>&1 & \
        echo $! > "${server_pid_file}")
    
    echo "Shared service started in background (PID: $(cat "${server_pid_file}"))."
    echo "Waiting a few seconds for the service to initialize..."
    sleep 5 # Increased sleep for service init

    if ! ps -p "$(cat "${server_pid_file}")" > /dev/null; then
        log_message "ERROR: Shared service failed to start for ${app_name}. Check ${server_log}"
        return 1
    fi

    log_message "Running Client Application for ${app_name} (Size: ${size}, Masking: ${masking_mode})..."
    # Client app (client_app.c) is updated to take --masking argument
    echo "Command: /usr/bin/time -v -o \"${result_prefix}_client.time\" gramine-sgx ${client_manifest} --masking ${masking_mode}"
    (cd "${CLIENT_ENCLAVE_DIR}" && \
        /usr/bin/time -v -o "${result_prefix}_client.time" \
        gramine-sgx "${client_manifest}" --masking "${masking_mode}" > "${client_log}" 2>&1)
    client_exit_code=$?

    log_message "Stopping Shared Enclave Service for ${app_name} (Masking: ${masking_mode})..."
    if [ -f "${server_pid_file}" ]; then
        kill "$(cat "${server_pid_file}")"
        wait "$(cat "${server_pid_file}")" 2>/dev/null || true
        rm -f "${server_pid_file}"
        echo "Shared service stopped."
    else
        echo "PID file not found, service might have already stopped or failed to start."
    fi

    if [ $client_exit_code -ne 0 ]; then
        log_message "ERROR: Client for ${app_name} (Size: ${size}) exited with code ${client_exit_code}."
        echo "Client log: ${client_log}"
        echo "Server log: ${server_log}"
        return 1
    fi
    log_message "Gramine SGX Shared Vector Add (Size: ${size}, Masking: ${masking_mode}) finished. Client log: ${client_log}"
    echo "Profiling hint (SGX): Use internal application timers. External perf/nvprof on gramine-sgx is complex."
    echo "Timing data (overall client process): ${result_prefix}_client.time"
    return $client_exit_code
}

run_native_onnx() {
    log_message "Running Native ONNX Inference (${ONNX_MODEL_NAME})"
    local mode="native"
    local app="onnx_inference"
    local result_prefix="${RESULTS_DIR}/${app}_${mode}_${TIMESTAMP}"
    local exec_path="${NATIVE_APPS_DIR}/bin/onnx_native_benchmark"
    local model_host_path="${NATIVE_APPS_DIR}/models/${ONNX_MODEL_NAME}" # Native app needs path to model

    check_executable "$exec_path" "Native ONNX Benchmark"
    if [ ! -f "$model_host_path" ]; then
        log_message "ERROR: ONNX Model not found at $model_host_path for native run."
        echo "Ensure it's placed there (e.g., copied from CI-Examples/shared-enclave/models/)"
        return 1
    fi
    
    log_message "Executing Native ONNX Inference. Log: ${result_prefix}.log, Time: ${result_prefix}.time"
    /usr/bin/time -v -o "${result_prefix}.time" "$exec_path" "$model_host_path" > "${result_prefix}.log" 2>&1
    cmd_exit_code=$?

    if [ $cmd_exit_code -ne 0 ]; then
        log_message "ERROR: Native ONNX Inference failed with exit code ${cmd_exit_code}."
        echo "See log: ${result_prefix}.log"
        return 1
    fi
    log_message "Native ONNX Inference finished. Results in ${result_prefix}.*"
    echo "Profiling hint: perf record -e cycles,instructions -g -- \"$exec_path\" \"$model_host_path\""
    echo "Profiling hint (GPU): sudo /usr/local/cuda/bin/nvprof --log-file \"${result_prefix}_nvprof.log\" \"$exec_path\" \"$model_host_path\""
    return 0
}

run_gramine_direct_onnx() {
    log_message "Running Gramine-Direct ONNX Inference (${ONNX_MODEL_NAME})"
    local mode="gramine_direct"
    local app="onnx_inference"
    local result_prefix="${RESULTS_DIR}/${app}_${mode}_${TIMESTAMP}"
    local exec_path_in_gramine="/benchmark_apps/onnx_native_benchmark"
    local model_path_in_gramine="/models/${ONNX_MODEL_NAME}" # Path inside Gramine, as per manifest mount
    local manifest_path="${NATIVE_APPS_DIR}/onnx_native.direct.manifest.template"

    check_executable "${NATIVE_APPS_DIR}/bin/onnx_native_benchmark" "Native ONNX Benchmark (for Gramine-Direct)"
    if [ ! -f "$manifest_path" ]; then
        log_message "ERROR: Direct manifest not found at $manifest_path"
        return 1
    fi
    # Ensure model is available for the manifest's relative mount path (./models)
    if [ ! -f "${NATIVE_APPS_DIR}/models/${ONNX_MODEL_NAME}" ]; then
        log_message "ERROR: ONNX Model not found at ${NATIVE_APPS_DIR}/models/${ONNX_MODEL_NAME} for gramine-direct run."
        echo "Ensure it's placed there. It should be mounted to $model_path_in_gramine by the manifest."
        return 1
    fi

    log_message "Executing Gramine-Direct ONNX Inference. Log: ${result_prefix}.log, Time: ${result_prefix}.time"
    /usr/bin/time -v -o "${result_prefix}.time" \
        gramine-direct "$manifest_path" "$model_path_in_gramine" > "${result_prefix}.log" 2>&1
    cmd_exit_code=$?

    if [ $cmd_exit_code -ne 0 ]; then
        log_message "ERROR: Gramine-Direct ONNX Inference failed with exit code ${cmd_exit_code}."
        echo "See log: ${result_prefix}.log"
        return 1
    fi
    log_message "Gramine-Direct ONNX Inference finished. Results in ${result_prefix}.*"
    echo "Profiling hint (Direct): perf record -e cycles,instructions -g -- gramine-direct \"$manifest_path\" \"$model_path_in_gramine\""
    echo "Profiling hint (GPU, Direct): sudo /usr/local/cuda/bin/nvprof --log-file \"${result_prefix}_nvprof.log\" gramine-direct \"$manifest_path\" \"$model_path_in_gramine\""
    return 0
}

run_sgx_shared_onnx() {
    local masking_mode="$1" # "aes_gcm" or "none"
    log_message "Running Gramine SGX Shared ONNX Inference (${ONNX_MODEL_NAME}, Masking: ${masking_mode})"
    
    local app_name="onnx_inference"
    local client_app_exec="client_app_onnx"
    local client_manifest="client-enclave-onnx.manifest.sgx"
    local result_prefix="${RESULTS_DIR}/${app_name}_sgx_shared_masking_${masking_mode}_${TIMESTAMP}"
    local server_log="${SHARED_ENCLAVE_DIR}/logs/${app_name}_server_sgx_${TIMESTAMP}.log" # Potentially shared server log
    local client_log="${CLIENT_ENCLAVE_DIR}/logs/${app_name}_client_sgx_masking_${masking_mode}_${TIMESTAMP}.log"
    local server_pid_file="${SHARED_ENCLAVE_DIR}/logs/server_${app_name}_sgx.pid"

    mkdir -p "${SHARED_ENCLAVE_DIR}/logs" "${CLIENT_ENCLAVE_DIR}/logs"

    if [ ! -f "${SHARED_ENCLAVE_DIR}/shared-enclave.manifest.sgx" ] || \
       [ ! -f "${CLIENT_ENCLAVE_DIR}/${client_manifest}" ]; then
        log_message "Signing SGX manifests for ${app_name}..."
        gramine-sgx-sign \
            --manifest "${SHARED_ENCLAVE_DIR}/shared-enclave.manifest.template" \
            --output "${SHARED_ENCLAVE_DIR}/shared-enclave.manifest.sgx" \
            --key "${SIGNER_KEY_FILE}"
        gramine-sgx-sign \
            --manifest "${CLIENT_ENCLAVE_DIR}/client-enclave-onnx.manifest.template" \
            --output "${CLIENT_ENCLAVE_DIR}/${client_manifest}" \
            --key "${SIGNER_KEY_FILE}"
        echo "Manifests signed."
    else
        echo "Manifests for ${app_name} (ONNX) already signed, skipping."
    fi
    gramine-sgx-sign \
        --manifest "${SHARED_ENCLAVE_DIR}/shared-enclave.manifest.template" \
        --output "${SHARED_ENCLAVE_DIR}/shared-enclave.manifest.sgx" \
        --key "${SIGNER_KEY_FILE}"
    # Assuming client_app_onnx uses a specific manifest if it's different
    gramine-sgx-sign \
        --manifest "${CLIENT_ENCLAVE_DIR}/client-enclave-onnx.manifest.template" \
        --output "${CLIENT_ENCLAVE_DIR}/client-enclave-onnx.manifest.sgx" \
        --key "${SIGNER_KEY_FILE}"
    echo "Manifests signed."
    
    log_message "Starting Shared Enclave Service for ${app_name}..."
    (cd "${SHARED_ENCLAVE_DIR}" && \
        gramine-sgx shared-enclave.manifest.sgx > "${server_log}" 2>&1 & \
        echo $! > "${server_pid_file}")
    
    echo "Shared service started (PID: $(cat "${server_pid_file}")). Waiting for init..."
    sleep 10 # ONNX model loading can take time

    if ! ps -p "$(cat "${server_pid_file}")" > /dev/null; then
        log_message "ERROR: Shared service failed to start for ${app_name}. Check ${server_log}"
        return 1
    fi

    log_message "Running Client Application for ${app_name} (Masking: ${masking_mode})..."
    echo "Command: /usr/bin/time -v -o \"${result_prefix}_client.time\" gramine-sgx ${client_manifest} --masking ${masking_mode}"
    (cd "${CLIENT_ENCLAVE_DIR}" && \
        /usr/bin/time -v -o "${result_prefix}_client.time" \
        gramine-sgx "${client_manifest}" --masking "${masking_mode}" > "${client_log}" 2>&1)
    client_exit_code=$?

    log_message "Stopping Shared Enclave Service for ${app_name} (Masking: ${masking_mode})..."
    if [ -f "${server_pid_file}" ]; then
        kill "$(cat "${server_pid_file}")"
        wait "$(cat "${server_pid_file}")" 2>/dev/null || true
        rm -f "${server_pid_file}"
        echo "Shared service stopped."
    else
        echo "PID file not found."
    fi

    if [ $client_exit_code -ne 0 ]; then
        log_message "ERROR: Client for ${app_name} exited with code ${client_exit_code}."
        echo "Client log: ${client_log}"
        echo "Server log: ${server_log}"
        return 1
    fi
    log_message "Gramine SGX Shared ONNX Inference (Masking: ${masking_mode}) finished. Client log: ${client_log}"
    echo "Timing data (overall client process): ${result_prefix}_client.time"
    return $client_exit_code
}

run_native_gemm() {
    log_message "Running Native GEMM (M=$1, N=$2, K=$3)"
    local M="$1" N="$2" K="$3"
    local mode="native"
    local app="gemm"
    local result_prefix="${RESULTS_DIR}/${app}_${mode}_M${M}_N${N}_K${K}_${TIMESTAMP}"
    local exec_path="${NATIVE_APPS_DIR}/bin/gemm_native_benchmark"

    check_executable "$exec_path" "Native GEMM Benchmark"

    log_message "Executing Native GEMM. Log: ${result_prefix}.log, Time: ${result_prefix}.time"
    /usr/bin/time -v -o "${result_prefix}.time" "$exec_path" "$M" "$N" "$K" > "${result_prefix}.log" 2>&1
    cmd_exit_code=$?

    if [ $cmd_exit_code -ne 0 ]; then
        log_message "ERROR: Native GEMM failed with exit code ${cmd_exit_code}."
        echo "See log: ${result_prefix}.log"
        return 1
    fi
    log_message "Native GEMM finished. Results in ${result_prefix}.*"
    echo "Profiling hint: perf record -e cycles,instructions -g -- \"$exec_path\" \"$M\" \"$N\" \"$K\""
    echo "Profiling hint (GPU): sudo /usr/local/cuda/bin/nvprof --log-file \"${result_prefix}_nvprof.log\" \"$exec_path\" \"$M\" \"$N\" \"$K\""
    return 0
}

run_gramine_direct_gemm() {
    log_message "Running Gramine-Direct GEMM (M=$1, N=$2, K=$3)"
    local M="$1" N="$2" K="$3"
    local mode="gramine_direct"
    local app="gemm"
    local result_prefix="${RESULTS_DIR}/${app}_${mode}_M${M}_N${N}_K${K}_${TIMESTAMP}"
    local exec_path_in_gramine="/benchmark_apps/gemm_native_benchmark"
    local manifest_path="${NATIVE_APPS_DIR}/gemm_native.direct.manifest.template"

    check_executable "${NATIVE_APPS_DIR}/bin/gemm_native_benchmark" "Native GEMM Benchmark (for Gramine-Direct)"
    if [ ! -f "$manifest_path" ]; then
        log_message "ERROR: Direct manifest not found at $manifest_path"
        return 1
    fi

    log_message "Executing Gramine-Direct GEMM. Log: ${result_prefix}.log, Time: ${result_prefix}.time"
    /usr/bin/time -v -o "${result_prefix}.time" \
        gramine-direct "$manifest_path" "$M" "$N" "$K" > "${result_prefix}.log" 2>&1
    cmd_exit_code=$?

    if [ $cmd_exit_code -ne 0 ]; then
        log_message "ERROR: Gramine-Direct GEMM failed with exit code ${cmd_exit_code}."
        echo "See log: ${result_prefix}.log"
        return 1
    fi
    log_message "Gramine-Direct GEMM finished. Results in ${result_prefix}.*"
    echo "Profiling hint (Direct): perf record -e cycles,instructions -g -- gramine-direct \"$manifest_path\" \"$M\" \"$N\" \"$K\""
    echo "Profiling hint (GPU, Direct): sudo /usr/local/cuda/bin/nvprof --log-file \"${result_prefix}_nvprof.log\" gramine-direct \"$manifest_path\" \"$M\" \"$N\" \"$K\""
    return 0
}

run_sgx_shared_gemm() {
    local masking_mode="$1" # "aes_gcm" or "none"
    local M="$2" N="$3" K="$4"
    log_message "Running Gramine SGX Shared GEMM (M=${M}, N=${N}, K=${K}, Masking: ${masking_mode})"
    
    local app_name="gemm"
    local client_app_exec="client_app_gemm"
    local client_manifest="client-enclave-gemm.manifest.sgx"
    local result_prefix="${RESULTS_DIR}/${app_name}_sgx_shared_masking_${masking_mode}_M${M}N${N}K${K}_${TIMESTAMP}"
    local server_log="${SHARED_ENCLAVE_DIR}/logs/${app_name}_server_sgx_${TIMESTAMP}.log" # Potentially shared
    local client_log="${CLIENT_ENCLAVE_DIR}/logs/${app_name}_client_sgx_masking_${masking_mode}_M${M}N${N}K${K}_${TIMESTAMP}.log"
    local server_pid_file="${SHARED_ENCLAVE_DIR}/logs/server_${app_name}_sgx.pid"

    mkdir -p "${SHARED_ENCLAVE_DIR}/logs" "${CLIENT_ENCLAVE_DIR}/logs"

    if [ ! -f "${SHARED_ENCLAVE_DIR}/shared-enclave.manifest.sgx" ] || \
       [ ! -f "${CLIENT_ENCLAVE_DIR}/${client_manifest}" ]; then
        log_message "Signing SGX manifests for ${app_name}..."
        gramine-sgx-sign \
            --manifest "${SHARED_ENCLAVE_DIR}/shared-enclave.manifest.template" \
            --output "${SHARED_ENCLAVE_DIR}/shared-enclave.manifest.sgx" \
            --key "${SIGNER_KEY_FILE}"
        gramine-sgx-sign \
            --manifest "${CLIENT_ENCLAVE_DIR}/client-enclave-gemm.manifest.template" \
            --output "${CLIENT_ENCLAVE_DIR}/${client_manifest}" \
            --key "${SIGNER_KEY_FILE}"
        echo "Manifests signed."
    else
        echo "Manifests for ${app_name} (GEMM) already signed, skipping."
    fi
    gramine-sgx-sign \
        --manifest "${SHARED_ENCLAVE_DIR}/shared-enclave.manifest.template" \
        --output "${SHARED_ENCLAVE_DIR}/shared-enclave.manifest.sgx" \
        --key "${SIGNER_KEY_FILE}"
    gramine-sgx-sign \
        --manifest "${CLIENT_ENCLAVE_DIR}/client-enclave-gemm.manifest.template" \
        --output "${CLIENT_ENCLAVE_DIR}/client-enclave-gemm.manifest.sgx" \
        --key "${SIGNER_KEY_FILE}"
    echo "Manifests signed."

    log_message "Starting Shared Enclave Service for ${app_name}..."
    (cd "${SHARED_ENCLAVE_DIR}" && \
        gramine-sgx shared-enclave.manifest.sgx > "${server_log}" 2>&1 & \
        echo $! > "${server_pid_file}")
    
    echo "Shared service started (PID: $(cat "${server_pid_file}")). Waiting for init..."
    sleep 5 # cuBLAS init is usually fast

    if ! ps -p "$(cat "${server_pid_file}")" > /dev/null; then
        log_message "ERROR: Shared service failed to start for ${app_name}. Check ${server_log}"
        return 1
    fi

    log_message "Running Client Application for ${app_name} (M=${M}, N=${N}, K=${K}, Masking: ${masking_mode})..."
    # Client app (client_app_gemm.c) is updated to take --masking argument
    echo "Command: /usr/bin/time -v -o \"${result_prefix}_client.time\" gramine-sgx ${client_manifest} --masking ${masking_mode}"
    (cd "${CLIENT_ENCLAVE_DIR}" && \
        /usr/bin/time -v -o "${result_prefix}_client.time" \
        gramine-sgx "${client_manifest}" --masking "${masking_mode}" > "${client_log}" 2>&1)
    client_exit_code=$?

    log_message "Stopping Shared Enclave Service for ${app_name} (Masking: ${masking_mode})..."
    if [ -f "${server_pid_file}" ]; then
        kill "$(cat "${server_pid_file}")"
        wait "$(cat "${server_pid_file}")" 2>/dev/null || true
        rm -f "${server_pid_file}"
        echo "Shared service stopped."
    else
        echo "PID file not found."
    fi

    if [ $client_exit_code -ne 0 ]; then
        log_message "ERROR: Client for ${app_name} (M${M}N${N}K${K}) exited with code ${client_exit_code}."
        echo "Client log: ${client_log}"
        echo "Server log: ${server_log}"
        return 1
    fi
    log_message "Gramine SGX Shared GEMM (Masking: ${masking_mode}) finished. Client log: ${client_log}"
    echo "Timing data (overall client process): ${result_prefix}_client.time"
    return $client_exit_code
}

# --- Main Execution Logic ---
main() {
    log_message "Starting GPU Application Benchmarking Script"
    setup_dirs

    if [ "$SIGNER_KEY_FILE" == "${SCRIPT_DIR}/YOUR_SIGNER_KEY.pem" ] || [ ! -f "$SIGNER_KEY_FILE" ]; then
        log_message "ERROR: SIGNER_KEY_FILE is not configured or key not found: '$SIGNER_KEY_FILE'"
        echo "Please edit this script and set SIGNER_KEY_FILE to your actual SGX private key."
        # exit 1 # Commented out for now to allow script structure review without a key
    fi
    echo "Using SGX Signer Key: $SIGNER_KEY_FILE (ensure this is correct)"


    # --- Build Applications (Informational) ---
    log_message "Build Check"
    echo "This script assumes all applications (native, client-enclave, shared-enclave) are pre-built."
    echo "If not, please build them using their respective Makefiles/Meson configurations."
    echo "E.g., 'make -C ${SCRIPT_DIR}' and 'make -C ${NATIVE_APPS_DIR}' (Makefile for native apps to be created)."
    echo ""

    # --- Vector Addition Benchmarks ---
    log_message "BENCHMARKING: Vector Addition"
    run_native_vector_add "$VECTOR_SIZE_ELEMENTS" && echo "Native Vector Add: OK" || echo "Native Vector Add: FAILED"
    run_gramine_direct_vector_add "$VECTOR_SIZE_ELEMENTS" && echo "Gramine-Direct Vector Add: OK" || echo "Gramine-Direct Vector Add: FAILED"
    log_message "Running SGX Shared Vector Add (AES-GCM Masking)..."
    run_sgx_shared_vector_add "aes_gcm" "$VECTOR_SIZE_ELEMENTS" && echo "SGX Shared Vector Add (AES-GCM): OK" || echo "SGX Shared Vector Add (AES-GCM): FAILED"
    log_message "Running SGX Shared Vector Add (No Masking)..."
    run_sgx_shared_vector_add "none" "$VECTOR_SIZE_ELEMENTS" && echo "SGX Shared Vector Add (None): OK" || echo "SGX Shared Vector Add (None): FAILED"
    
    # --- ONNX Inference Benchmarks ---
    log_message "BENCHMARKING: ONNX Inference (MobileNetV2)"
    # Ensure the model file is available for native and direct modes
    # The native and direct manifests assume the model is at ./models/mobilenetv2-7.onnx relative to NATIVE_APPS_DIR
    # The shared enclave manifest assumes it's at ./models/mobilenetv2-7.onnx relative to SHARED_ENCLAVE_DIR
    # For simplicity, copy it to native-apps/models if it's not there.
    if [ ! -f "${NATIVE_APPS_DIR}/models/${ONNX_MODEL_NAME}" ]; then
        log_message "Copying ONNX model to ${NATIVE_APPS_DIR}/models/ for native/direct runs..."
        mkdir -p "${NATIVE_APPS_DIR}/models"
        if [ -f "${SHARED_ENCLAVE_DIR}/models/${ONNX_MODEL_NAME}" ]; then
            cp "${SHARED_ENCLAVE_DIR}/models/${ONNX_MODEL_NAME}" "${NATIVE_APPS_DIR}/models/"
        else
            log_message "WARNING: ONNX Model ${ONNX_MODEL_NAME} not found in shared enclave dir to copy for native tests."
            echo "Please ensure ${NATIVE_APPS_DIR}/models/${ONNX_MODEL_NAME} exists for native/direct ONNX tests."
        fi
    fi
    run_native_onnx && echo "Native ONNX: OK" || echo "Native ONNX: FAILED"
    run_gramine_direct_onnx && echo "Gramine-Direct ONNX: OK" || echo "Gramine-Direct ONNX: FAILED"
    log_message "Running SGX Shared ONNX Inference (AES-GCM Masking)..."
    run_sgx_shared_onnx "aes_gcm" && echo "SGX Shared ONNX (AES-GCM): OK" || echo "SGX Shared ONNX (AES-GCM): FAILED"
    log_message "Running SGX Shared ONNX Inference (No Masking)..."
    run_sgx_shared_onnx "none" && echo "SGX Shared ONNX (None): OK" || echo "SGX Shared ONNX (None): FAILED"

    # --- GEMM (cuBLAS) Benchmarks ---
    log_message "BENCHMARKING: GEMM (cuBLAS SGEMM)"
    run_native_gemm "$GEMM_M" "$GEMM_N" "$GEMM_K" && echo "Native GEMM: OK" || echo "Native GEMM: FAILED"
    run_gramine_direct_gemm "$GEMM_M" "$GEMM_N" "$GEMM_K" && echo "Gramine-Direct GEMM: OK" || echo "Gramine-Direct GEMM: FAILED"
    log_message "Running SGX Shared GEMM (AES-GCM Masking)..."
    run_sgx_shared_gemm "aes_gcm" "$GEMM_M" "$GEMM_N" "$GEMM_K" && echo "SGX Shared GEMM (AES-GCM): OK" || echo "SGX Shared GEMM (AES-GCM): FAILED"
    log_message "Running SGX Shared GEMM (No Masking)..."
    run_sgx_shared_gemm "none" "$GEMM_M" "$GEMM_N" "$GEMM_K" && echo "SGX Shared GEMM (None): OK" || echo "SGX Shared GEMM (None): FAILED"

    log_message "Benchmarking Script Finished"
    echo "Results and logs are stored in: ${RESULTS_DIR}"
    echo "Summary of time files (overall process execution):"
    grep "Elapsed (wall clock) time" "${RESULTS_DIR}"/*.time || echo "No .time files found or they lack expected 'Elapsed' time string."
    echo ""
    echo "Detailed application-internal GPU timings should be in the .log files for each run."
}

# --- Script Entry Point ---
main "$@"
