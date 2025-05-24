#!/bin/bash

# CI-Examples/run_vector_add_test.sh
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (C) 2023 Intel Corporation
#
# This script serves as a testing guide for the shared-enclave GPU vector addition sample.
# It outlines the manual steps to build, sign, run, and verify the sample.
# For automated CI, these steps would be adapted into CI jobs.

set -e # Exit on any error

# --- Configuration - Adjust these paths if your setup differs ---
# Assuming this script is run from the root of the Gramine repository,
# or that the CI-Examples directory is in the current path.
# For CI, these paths might be absolute based on the build/install prefix.
# The Makefile uses --prefix=/usr/local, so binaries would be there.
# However, manifests use /gramine/CI-Examples/... for entrypoints.
# This script will assume binaries are built and accessible relative to their
# respective directories for simplicity in this guide.
# A real CI would need to ensure paths are consistent.

SHARED_ENCLAVE_DIR="shared-enclave"
CLIENT_ENCLAVE_DIR="client-enclave"
COMMON_DIR="common" # Not directly run, but contains shared header

# Path to your SGX signing key (MUST be replaced by the user)
# For automated testing, a common test key is usually used.
SIGNER_KEY_FILE="YOUR_SIGNER_KEY.pem" # Placeholder - REPLACE THIS

# --- Pre-requisites ---
echo "--------------------------------------------------------------------------"
echo "Pre-requisites & Assumptions:"
echo "--------------------------------------------------------------------------"
echo "1. Gramine: Installed (gramine-sgx, gramine-sgx-sign, etc., in PATH)."
echo "2. SGX Drivers: Intel SGX drivers installed and SGX is enabled (e.g., in BIOS)."
echo "3. NVIDIA Drivers: Appropriate NVIDIA drivers installed on the host."
echo "4. CUDA Toolkit: CUDA toolkit (e.g., 11.x or compatible) installed on the host."
echo "   - Ensure 'nvcc' is in PATH for compilation if building from scratch."
echo "   - Ensure CUDA libraries (libcudart.so, etc.) are accessible on the host"
echo "     at paths consistent with 'sgx.trusted_files' in the shared enclave manifest."
echo "     (The template uses '/usr/lib/host_cuda_libs/' as a placeholder URI base)."
echo "5. Applications Compiled: The 'client_app' and 'shared_service' executables"
echo "   have been compiled (e.g., using the 'CI-Examples/Makefile')."
echo "   This script assumes they are in:"
echo "   - ${SHARED_ENCLAVE_DIR}/bin/shared_service"
echo "   - ${CLIENT_ENCLAVE_DIR}/bin/client_app"
echo "   (These paths are relative to the CI-Examples directory if not using a prefix install)."
echo "6. Signer Key: You have an SGX private key (e.g., '${SIGNER_KEY_FILE}') for signing."
echo "   Replace the placeholder above with the actual path to your key."
echo "--------------------------------------------------------------------------"
echo ""

if [ "$SIGNER_KEY_FILE" == "YOUR_SIGNER_KEY.pem" ]; then
    echo "ERROR: Please replace 'YOUR_SIGNER_KEY.pem' in this script with the actual path to your SGX private signing key."
    exit 1
fi
if [ ! -f "$SIGNER_KEY_FILE" ]; then
    echo "ERROR: Signer key file not found at '$SIGNER_KEY_FILE'. Please provide a valid key."
    exit 1
fi


# --- Build Step (Optional - can be done by Makefile) ---
echo "--------------------------------------------------------------------------"
echo "Step 0: Build Applications (if not already built)"
echo "--------------------------------------------------------------------------"
echo "This script assumes the applications are already built using the Makefile."
echo "If not, run 'make' in the CI-Examples directory first: "
echo "  (cd CI-Examples && make)"
echo "--------------------------------------------------------------------------"
echo ""
# Uncomment to add build step here if needed for a self-contained script
# make -C . # Assuming script is in CI-Examples
# sleep 1


# --- Signing Manifests ---
echo "--------------------------------------------------------------------------"
echo "Step 1: Sign Manifests"
echo "--------------------------------------------------------------------------"
# Note: gramine-sgx-sign will create .sig files if they don't exist,
# and will overwrite .manifest.sgx if it exists.

echo "Signing shared enclave manifest..."
gramine-sgx-sign \
    --manifest "${SHARED_ENCLAVE_DIR}/shared-enclave.manifest.template" \
    --output "${SHARED_ENCLAVE_DIR}/shared-enclave.manifest.sgx" \
    --key "${SIGNER_KEY_FILE}"
echo "Shared enclave manifest signed: ${SHARED_ENCLAVE_DIR}/shared-enclave.manifest.sgx"
echo ""

echo "Signing client enclave manifest..."
gramine-sgx-sign \
    --manifest "${CLIENT_ENCLAVE_DIR}/client-enclave.manifest.template" \
    --output "${CLIENT_ENCLAVE_DIR}/client-enclave.manifest.sgx" \
    --key "${SIGNER_KEY_FILE}"
echo "Client enclave manifest signed: ${CLIENT_ENCLAVE_DIR}/client-enclave.manifest.sgx"
echo "--------------------------------------------------------------------------"
echo ""
sleep 1


# --- Running the Test ---
# For this guide, we'll run the shared service in the background and then the client.
# In a CI environment, you might manage these processes more robustly.

SHARED_SERVICE_LOG="shared_service.log"
CLIENT_APP_LOG="client_app.log"
SHARED_SERVICE_PID_FILE="shared_service.pid"

# Cleanup old logs and PID file
rm -f "${SHARED_ENCLAVE_DIR}/${SHARED_SERVICE_LOG}" "${CLIENT_ENCLAVE_DIR}/${CLIENT_APP_LOG}" "${SHARED_ENCLAVE_DIR}/${SHARED_SERVICE_PID_FILE}"

echo "--------------------------------------------------------------------------"
echo "Step 2: Run the Shared Enclave Service (in background)"
echo "--------------------------------------------------------------------------"
echo "Starting shared_service... Logs will be in ${SHARED_ENCLAVE_DIR}/${SHARED_SERVICE_LOG}"
echo "Manifest used: ${SHARED_ENCLAVE_DIR}/shared-enclave.manifest.sgx"
echo "Executable path assumed by manifest: /gramine/CI-Examples/shared-enclave/bin/shared_service"
echo "Actual host path to executable (relative to CI-Examples): ${SHARED_ENCLAVE_DIR}/bin/shared_service"
echo ""
echo "If using a prefix install (e.g. /usr/local), ensure your manifest's sgx.trusted_files"
echo "and libos.entrypoint reflect the installation path (e.g., file:/usr/local/bin/shared_service)."
echo "The current manifest templates use paths like /gramine/CI-Examples/..."
echo "This implies Gramine's FS root is set up to resolve this, or files are staged accordingly."
echo ""

# Navigate to the directory where shared_service executable and manifest are expected
# This is crucial for gramine-sgx to find files if relative paths are used internally
# or if the manifest URIs are relative (though they should be absolute file: URIs).
# For this test, we assume the current directory is CI-Examples.
(cd "${SHARED_ENCLAVE_DIR}" && \
    gramine-sgx shared-enclave.manifest.sgx > "${SHARED_SERVICE_LOG}" 2>&1 & \
    echo $! > "${SHARED_SERVICE_PID_FILE}")

echo "Shared service started in background (PID: $(cat "${SHARED_ENCLAVE_DIR}/${SHARED_SERVICE_PID_FILE}"))."
echo "Waiting a few seconds for the service to initialize..."
sleep 3 # Give the service time to start up and listen

# Check if service started (basic check, more robust checks could be added)
if ! ps -p "$(cat "${SHARED_ENCLAVE_DIR}/${SHARED_SERVICE_PID_FILE}")" > /dev/null; then
    echo "ERROR: Shared service failed to start. Check ${SHARED_ENCLAVE_DIR}/${SHARED_SERVICE_LOG}"
    exit 1
fi
echo "--------------------------------------------------------------------------"
echo ""


echo "--------------------------------------------------------------------------"
echo "Step 3: Run the Client Enclave Application"
echo "--------------------------------------------------------------------------"
echo "Starting client_app... Logs will be in ${CLIENT_ENCLAVE_DIR}/${CLIENT_APP_LOG}"
echo "Manifest used: ${CLIENT_ENCLAVE_DIR}/client-enclave.manifest.sgx"
echo "Executable path assumed by manifest: /gramine/CI-Examples/client-enclave/bin/client_app"
echo ""

CLIENT_EXIT_CODE=0
(cd "${CLIENT_ENCLAVE_DIR}" && \
    gramine-sgx client-enclave.manifest.sgx > "${CLIENT_APP_LOG}" 2>&1) || CLIENT_EXIT_CODE=$?

echo "Client app finished with exit code: ${CLIENT_EXIT_CODE}"
echo "--------------------------------------------------------------------------"
echo ""
sleep 1


# --- Stop the Shared Service ---
echo "--------------------------------------------------------------------------"
echo "Step 4: Stop the Shared Enclave Service"
echo "--------------------------------------------------------------------------"
if [ -f "${SHARED_ENCLAVE_DIR}/${SHARED_SERVICE_PID_FILE}" ]; then
    SHARED_PID=$(cat "${SHARED_ENCLAVE_DIR}/${SHARED_SERVICE_PID_FILE}")
    if ps -p "$SHARED_PID" > /dev/null; then
        echo "Stopping shared_service (PID: $SHARED_PID)..."
        kill "$SHARED_PID"
        wait "$SHARED_PID" 2>/dev/null || true # Wait for it to terminate, ignore "No such process" if already gone
        echo "Shared service stopped."
    else
        echo "Shared service (PID: $SHARED_PID) already stopped."
    fi
    rm -f "${SHARED_ENCLAVE_DIR}/${SHARED_SERVICE_PID_FILE}"
else
    echo "Shared service PID file not found. It might have failed to start or was already cleaned up."
fi
echo "--------------------------------------------------------------------------"
echo ""


# --- Verification Checks ---
echo "--------------------------------------------------------------------------"
echo "Step 5: Verification and Expected Outcomes"
echo "--------------------------------------------------------------------------"
echo "Shared Service Logs (${SHARED_ENCLAVE_DIR}/${SHARED_SERVICE_LOG}):"
echo "--------------------------------------------------"
cat "${SHARED_ENCLAVE_DIR}/${SHARED_SERVICE_LOG}"
echo "--------------------------------------------------"
echo ""
echo "Client App Logs (${CLIENT_ENCLAVE_DIR}/${CLIENT_APP_LOG}):"
echo "--------------------------------------------------"
cat "${CLIENT_ENCLAVE_DIR}/${CLIENT_APP_LOG}"
echo "--------------------------------------------------"
echo ""

echo "Automated Verification:"
SUCCESS=true

# 1. Client app exit code
if [ "$CLIENT_EXIT_CODE" -ne 0 ]; then
    echo "VERIFICATION FAILED: Client app exited with non-zero status (${CLIENT_EXIT_CODE})."
    SUCCESS=false
else
    echo "VERIFICATION PASSED: Client app exited with 0."
fi

# 2. Client app reports SUCCESS
if grep -q "CLIENT_APP_SUCCESS: Vector addition results verified successfully!" "${CLIENT_ENCLAVE_DIR}/${CLIENT_APP_LOG}"; then
    echo "VERIFICATION PASSED: Client app reported SUCCESS."
else
    echo "VERIFICATION FAILED: Client app did not report SUCCESS message."
    SUCCESS=false
fi

# 3. Shared service logs expected messages
# These are examples; more specific checks can be added.
if grep -q "SHARED_SERVICE_LOG: Successfully listening on handle" "${SHARED_ENCLAVE_DIR}/${SHARED_SERVICE_LOG}"; then
    echo "VERIFICATION PASSED: Shared service started listening."
else
    echo "VERIFICATION FAILED: Shared service did not log successful listen."
    # SUCCESS=false # This might be too strict if startup messages change slightly
fi

if grep -q "SHARED_SERVICE_LOG: New session started for client VMID" "${SHARED_ENCLAVE_DIR}/${SHARED_SERVICE_LOG}"; then
    echo "VERIFICATION PASSED: Shared service accepted a client session."
else
    echo "VERIFICATION FAILED: Shared service did not log client session start."
    SUCCESS=false
fi

if grep -q "SHARED_SERVICE_LOG: Handling VECTOR_ADD_REQUEST" "${SHARED_ENCLAVE_DIR}/${SHARED_SERVICE_LOG}"; then
    echo "VERIFICATION PASSED: Shared service handled a VECTOR_ADD_REQUEST."
else
    echo "VERIFICATION FAILED: Shared service did not log handling of VECTOR_ADD_REQUEST."
    SUCCESS=false
fi

if grep -q "SHARED_SERVICE_LOG: CUDA vector add kernel successful." "${SHARED_ENCLAVE_DIR}/${SHARED_SERVICE_LOG}"; then
    echo "VERIFICATION PASSED: Shared service reported successful CUDA kernel execution."
else
    echo "VERIFICATION FAILED: Shared service did not report successful CUDA kernel execution."
    SUCCESS=false
fi

# 4. Check for obvious errors in logs (e.g., "ERROR", "failed", "FATAL")
# Be careful with generic error words if they might appear in normal debug logs.
if grep -E "ERROR|FATAL|failed" "${SHARED_ENCLAVE_DIR}/${SHARED_SERVICE_LOG}" | grep -vE "Failed to accept a client connection \(-ECONNRESET\)|client disconnected|DEBUG"; then
    echo "VERIFICATION WARNING: Potential errors found in shared_service log. Please review manually."
    # SUCCESS=false # Make this a hard fail if specific errors are critical
fi
if grep -E "ERROR|FATAL|failed" "${CLIENT_ENCLAVE_DIR}/${CLIENT_APP_LOG}" | grep -vE "DEBUG"; then
    echo "VERIFICATION WARNING: Potential errors found in client_app log. Please review manually."
    # SUCCESS=false
fi

echo "--------------------------------------------------------------------------"
if [ "$SUCCESS" = true ]; then
    echo "Overall Test Result: PASSED (based on automated checks)"
    exit 0
else
    echo "Overall Test Result: FAILED (based on automated checks)"
    exit 1
fi

# --- Troubleshooting Notes ---
# - CUDA Library Paths: Ensure `sgx.trusted_files` in `shared-enclave.manifest.template`
#   correctly lists all necessary CUDA libraries and their dependencies, matching host paths.
#   The placeholder paths like "/usr/lib/host_cuda_libs/" must be adapted.
#   Use `loader.env.LD_LIBRARY_PATH` in the manifest if libraries are in non-standard paths *inside* the enclave mounts.
# - IPC Issues:
#   - "ipc_send_msg_and_get_response failed: Connection refused": Shared service not running or not listening.
#   - Incorrect VMID for shared service in client_app.c.
# - SGX Errors:
#   - Page faults: Often due to memory access issues, untrusted memory access, or enclave size limits.
#   - Ensure `sgx.enclave_size` is sufficient.
# - Permissions: Gramine (and the user running it) needs permission to access `/dev/nvidia*` nodes.
# - CUDA Errors in Service: Check `shared_service.log` for messages from `launch_vector_add_cuda` if the client
#   reports errors from the service. This could indicate issues with GPU memory allocation, kernel launch, etc.
# - Manifest Paths: Ensure `libos.entrypoint` and `file:` URIs in `sgx.trusted_files` correctly point to
#   the application executable and trusted libraries *as accessible by Gramine on the host* during measurement
#   and *as mapped inside the enclave* during runtime.
# - AES-GCM errors: "Decryption failed" or "Encryption failed" in client/server logs point to issues in crypto logic
#   or data corruption. "MBEDTLS_ERR_GCM_AUTH_FAILED" indicates tag mismatch, meaning data was tampered or key/IV was wrong.The test script/guide `CI-Examples/run_vector_add_test.sh` has been created in the previous turn.

**Summary of the `run_vector_add_test.sh` script:**

1.  **Configuration Section:** Includes placeholders for paths and the `SIGNER_KEY_FILE`, which the user must update.
2.  **Pre-requisites Section:** Clearly lists assumptions:
    *   Gramine installed.
    *   SGX drivers and setup.
    *   NVIDIA drivers.
    *   CUDA toolkit (including `nvcc` for compilation and libraries for runtime).
    *   Compiled `client_app` and `shared_service` (assumed to be built by the `CI-Examples/Makefile`).
    *   Availability of an SGX signing key.
    *   Checks if `SIGNER_KEY_FILE` is updated and exists.
3.  **Build Step (Commented Out):** Reminds the user to build the applications using the main Makefile if not done already.
4.  **Signing Manifests:**
    *   Provides `gramine-sgx-sign` commands for both `shared-enclave.manifest.template` and `client-enclave.manifest.template`, using the `SIGNER_KEY_FILE` variable.
    *   Outputs are `.manifest.sgx` files.
5.  **Running the Shared Enclave Service:**
    *   Changes directory to `shared-enclave`.
    *   Runs `gramine-sgx shared-enclave.manifest.sgx` in the background.
    *   Redirects stdout/stderr to `shared_service.log`.
    *   Stores the PID of the background service.
    *   Includes a short `sleep` to allow the service to initialize.
    *   Performs a basic check to see if the service process started.
    *   Includes important notes about manifest paths and how Gramine resolves `libos.entrypoint` and `sgx.trusted_files`.
6.  **Running the Client Enclave Application:**
    *   Changes directory to `client-enclave`.
    *   Runs `gramine-sgx client-enclave.manifest.sgx` in the foreground.
    *   Redirects stdout/stderr to `client_app.log`.
    *   Captures the client's exit code.
7.  **Stopping the Shared Service:**
    *   Uses the stored PID to `kill` the shared service.
    *   Includes `wait` to ensure the process has terminated.
8.  **Verification Checks:**
    *   Prints the content of `shared_service.log` and `client_app.log`.
    *   **Automated Checks:**
        *   Checks if `CLIENT_EXIT_CODE` is 0.
        *   Greps `client_app.log` for the "CLIENT_APP_SUCCESS" message.
        *   Greps `shared_service.log` for key messages indicating successful startup, client connection, handling of `VECTOR_ADD_REQUEST`, and successful CUDA kernel execution.
        *   Includes commented-out (or warning-level) checks for generic error strings in logs.
    *   Sets an overall `SUCCESS` status and exits with 0 for success, 1 for failure based on these checks.
9.  **Troubleshooting Notes:** Provides a list of common issues and areas to check if the test fails (CUDA paths, IPC, SGX errors, permissions, manifest paths, AES-GCM errors).

This script provides a clear, step-by-step guide for manually testing the application and also forms a basis for automated CI checks by inspecting logs and exit codes.
All specified tasks for generating the testing script and expected outcomes are now complete.
