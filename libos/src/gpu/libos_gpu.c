// SPDX-License-Identifier: GPL-3.0-only OR BSD-3-Clause
/*
 * Copyright (C) 2023 Gramine contributors
 *
 * This file is part of the Gramine project.
 */

#include "libos_internal.h"
#include "libos_gpu.h"
#include "libos_gpu_ipc.h" // Added for IPC structures
#include "log.h"
#include <errno.h>
#include <string.h> // For memcpy, strlen, memset
#include "pal_api.h" // For PAL_HANDLE, PalStreamOpen, PalStreamWrite, PalStreamRead, PalObjectDestroy, PalRandomBitsRead

#include <mbedtls/gcm.h>
#include <mbedtls/error.h> // For mbedtls_strerror

// --- Global variables for IPC and Shared Memory (Enclave Side) ---
static PAL_HANDLE g_gpu_proxy_pipe_enclave_to_proxy = PAL_HANDLE_INITIALIZER;
static PAL_HANDLE g_gpu_proxy_pipe_proxy_to_enclave = PAL_HANDLE_INITIALIZER;
static void* g_gpu_shared_mem = NULL; // Points to the start of the SHM region
static size_t g_gpu_shm_size = 0;     // Total size of the SHM region

static uint8_t g_gpu_session_key[32];
static bool g_session_key_initialized = false;
static mbedtls_gcm_context g_aes_gcm_ctx;
static uint64_t g_encryption_invocation_count = 0; // For IV generation

// Placeholder for SHM path, actual value might come from manifest or be fixed
// TODO: Read SHM path from manifest: sgx.mounts item with path="/dev/gramine_gpu_shm" and type="untrusted_shm"
#define GPU_SHM_PATH_INTERNAL "/dev/gramine_gpu_shm" // Default internal path seen by LibOS

// Placeholder for pipe paths
// TODO: Read pipe base name from manifest option sgx.gpu_ipc_pipe_name (e.g., "gramine_gpu_ipc")
//       and construct full paths like "/tmp/gramine_gpu_ipc_e2p" and "/tmp/gramine_gpu_ipc_p2e".
//       For now, using fixed paths.
#define GPU_PIPE_ENCLAVE_TO_PROXY "pipe:/tmp/gramine_gpu_ipc_e2p"
#define GPU_PIPE_PROXY_TO_ENCLAVE "pipe:/tmp/gramine_gpu_ipc_p2e"


// Helper function to send a command and receive a response
static int send_command_receive_response(gpu_message_header_t* req_header, void* req_payload,
                                         gpu_message_header_t* resp_header, void* resp_payload, uint32_t max_resp_payload_size) {
    ssize_t bytes_written, bytes_read;

    // Send request header
    bytes_written = PalStreamWrite(g_gpu_proxy_pipe_enclave_to_proxy, req_header, sizeof(gpu_message_header_t), NULL, -1);
    if (bytes_written != sizeof(gpu_message_header_t)) {
        log_error("Failed to write request header to proxy pipe: %ld", bytes_written);
        return -EIO;
    }

    // Send request payload (if any)
    if (req_header->payload_size > 0 && req_payload) {
        bytes_written = PalStreamWrite(g_gpu_proxy_pipe_enclave_to_proxy, req_payload, req_header->payload_size, NULL, -1);
        if (bytes_written != (ssize_t)req_header->payload_size) {
            log_error("Failed to write request payload to proxy pipe: %ld", bytes_written);
            return -EIO;
        }
    }

    // Read response header
    bytes_read = PalStreamRead(g_gpu_proxy_pipe_proxy_to_enclave, resp_header, sizeof(gpu_message_header_t), NULL, -1);
    if (bytes_read != sizeof(gpu_message_header_t)) {
        log_error("Failed to read response header from proxy pipe: %ld", bytes_read);
        return -EIO;
    }

    // Read response payload (if any)
    if (resp_header->payload_size > 0 && resp_payload) {
        if (resp_header->payload_size > max_resp_payload_size) {
            log_error("Response payload size (%u) exceeds max allowed (%u)", resp_header->payload_size, max_resp_payload_size);
            return -ENOMEM; // Or some other error
        }
        bytes_read = PalStreamRead(g_gpu_proxy_pipe_proxy_to_enclave, resp_payload, resp_header->payload_size, NULL, -1);
        if (bytes_read != (ssize_t)resp_header->payload_size) {
            log_error("Failed to read response payload from proxy pipe: %ld", bytes_read);
            return -EIO;
        }
    }
    return resp_header->status; // Return status from proxy
}


int gramine_cuda_init(void) {
    // TODO: Check sgx.gramine_gpu_enable from manifest before proceeding.
    log_debug("gramine_cuda_init called");

    // 1. Get shared memory details (path/ID, size) from manifest.
    //    The manifest should define an 'untrusted_shm' mount with a specific path, e.g., /dev/gramine_gpu_shm.
    //    The LibOS needs to find this mount to get its host-backed URI and size.
    //    For now, assume fixed path (GPU_SHM_PATH_INTERNAL) and size.
    //    Example: g_gpu_shm_size = get_manifest_shm_size_for_path(GPU_SHM_PATH_INTERNAL);
    g_gpu_shm_size = 4 * 1024 * 1024; // Example: 4MB SHM region, TODO: Get from manifest

    // 2. Use PAL calls to map/access this shared memory.
    //    The actual mapping depends on how Gramine exposes `untrusted_shm` mounts.
    //    If Gramine pre-maps it and provides a handle or direct pointer:
    //        g_gpu_shared_mem = get_premapped_shm_ptr(GPU_SHM_PATH_INTERNAL);
    //    If LibOS needs to map it using a PAL device associated with the untrusted_shm path:
    //        PalDeviceMap(GPU_SHM_PATH_INTERNAL, NULL, g_gpu_shm_size, PAL_PROT_READ | PAL_PROT_WRITE, PAL_MAP_SHARED, &g_gpu_shared_mem);
    //    For now, we'll skip the actual mapping and assume g_gpu_shared_mem will be populated by a future PAL call.
    log_debug("SHM setup: internal_path=%s, size=%zu (placeholder, actual mapping & size from manifest TBD)",
              GPU_SHM_PATH_INTERNAL, g_gpu_shm_size);
    // TODO: Implement actual SHM mapping based on manifest configuration and PAL capabilities for untrusted_shm.
    //       g_gpu_shared_mem should be populated here.
    //       If mapping fails, return an error.
    // For simulation if SHM not truly mapped:
    if (g_gpu_shm_size > 0 && !g_gpu_shared_mem) {
         log_warning("Actual SHM mapping via PAL calls is not yet implemented in gramine_cuda_init. g_gpu_shared_mem is NULL.");
         // To allow testing of IPC logic without full SHM, one might allocate a temporary buffer:
         // g_gpu_shared_mem = malloc(g_gpu_shm_size); // DANGER: This is NOT shared memory and only for isolated testing.
         // If using such a malloc, ensure it's freed in gramine_cuda_shutdown.
    }


    // 3. Open PAL pipes to the proxy.
    //    The proxy executable path (`sgx.gpu_proxy_executable`) is used by PAL to launch the proxy.
    //    The pipe names themselves might be derived from a manifest option or fixed.
    //    TODO: Launch the proxy using `sgx.gpu_proxy_executable` if not already running,
    //          potentially via a PAL call like `PalLaunchProxy(path_from_manifest, ...)`
    //          This call might also handle pipe creation and return handles.
    //    Using two unidirectional pipes for clearer req/resp flow.
    int ret = PalStreamOpen(GPU_PIPE_ENCLAVE_TO_PROXY, PAL_ACCESS_RDWR, 0, &g_gpu_proxy_pipe_enclave_to_proxy);
    if (ret) {
        log_error("Failed to open pipe to proxy (enclave_to_proxy): %d", ret);
        return ret;
    }
    log_debug("Opened pipe: %s", GPU_PIPE_ENCLAVE_TO_PROXY);

    ret = PalStreamOpen(GPU_PIPE_PROXY_TO_ENCLAVE, PAL_ACCESS_RDWR, 0, &g_gpu_proxy_pipe_proxy_to_enclave);
    if (ret) {
        log_error("Failed to open pipe from proxy (proxy_to_enclave): %d", ret);
        PalObjectDestroy(&g_gpu_proxy_pipe_enclave_to_proxy);
        return ret;
    }
    log_debug("Opened pipe: %s", GPU_PIPE_PROXY_TO_ENCLAVE);

    // 4. Send a CMD_INIT message to the proxy.
    gpu_message_header_t req_header, resp_header;
    cmd_init_req_t init_req;
    cmd_init_resp_t init_resp;

    // Generate session key
    ret = PalRandomBitsRead(g_gpu_session_key, sizeof(g_gpu_session_key));
    if (ret) {
        log_error("Failed to generate session key: %d", ret);
        PalObjectDestroy(&g_gpu_proxy_pipe_enclave_to_proxy);
        PalObjectDestroy(&g_gpu_proxy_pipe_proxy_to_enclave);
        return ret;
    }

    // Initialize mbedTLS GCM context with the generated key
    mbedtls_gcm_init(&g_aes_gcm_ctx);
    ret = mbedtls_gcm_setkey(&g_aes_gcm_ctx, MBEDTLS_CIPHER_ID_AES, g_gpu_session_key, 256);
    if (ret != 0) {
        char error_buf[100];
        mbedtls_strerror(ret, error_buf, sizeof(error_buf));
        log_error("mbedtls_gcm_setkey failed: %s", error_buf);
        mbedtls_gcm_free(&g_aes_gcm_ctx);
        PalObjectDestroy(&g_gpu_proxy_pipe_enclave_to_proxy);
        PalObjectDestroy(&g_gpu_proxy_pipe_proxy_to_enclave);
        memset(g_gpu_session_key, 0, sizeof(g_gpu_session_key)); // Clear key
        return -ret; // mbedTLS errors are typically negative
    }
    g_session_key_initialized = true;
    log_debug("Session key generated and mbedTLS GCM context initialized.");

    memcpy(init_req.session_key, g_gpu_session_key, sizeof(g_gpu_session_key));
    // Send the *internal* path of the SHM mount to the proxy.
    strncpy(init_req.shm_path, GPU_SHM_PATH_INTERNAL, MAX_SHM_PATH_LEN -1);
    init_req.shm_path[MAX_SHM_PATH_LEN -1] = '\0';
    init_req.shm_size = g_gpu_shm_size; // Send the actual (or configured) size

    req_header.type = CMD_INIT;
    req_header.payload_size = sizeof(cmd_init_req_t);
    req_header.status = 0; // Not used for requests from enclave

    log_debug("Sending CMD_INIT to proxy. SHM path: %s, size: %zu", init_req.shm_path, init_req.shm_size);

    ret = send_command_receive_response(&req_header, &init_req, &resp_header, &init_resp, sizeof(cmd_init_resp_t));
    if (ret != 0 || resp_header.type != CMD_INIT || resp_header.status != 0) {
        log_error("CMD_INIT failed. Proxy ret: %d, response status: %d, type: %d", ret, resp_header.status, resp_header.type);
        PalObjectDestroy(&g_gpu_proxy_pipe_enclave_to_proxy);
        PalObjectDestroy(&g_gpu_proxy_pipe_proxy_to_enclave);
        mbedtls_gcm_free(&g_aes_gcm_ctx);
        memset(g_gpu_session_key, 0, sizeof(g_gpu_session_key));
        g_session_key_initialized = false;
        // if (g_gpu_shared_mem) { /* TODO: Unmap SHM */ }
        return (ret != 0) ? ret : (resp_header.status != 0 ? resp_header.status : -EIO);
    }

    log_debug("CMD_INIT successful. Proxy response status_report: %d", init_resp.proxy_status_report);
    return 0;
}

int gramine_cuda_shutdown(void) {
    log_debug("gramine_cuda_shutdown called");

    if (!PalHandleIsValid(g_gpu_proxy_pipe_enclave_to_proxy)) {
        log_warning("Shutdown called but pipe to proxy is not valid.");
    } else {
        gpu_message_header_t req_header; // No response expected for shutdown
        req_header.type = CMD_SHUTDOWN;
        req_header.payload_size = 0;
        req_header.status = 0;

        log_debug("Sending CMD_SHUTDOWN to proxy.");
        ssize_t bytes_written = PalStreamWrite(g_gpu_proxy_pipe_enclave_to_proxy, &req_header, sizeof(gpu_message_header_t), NULL, -1);
        if (bytes_written != sizeof(gpu_message_header_t)) {
            log_error("Failed to write CMD_SHUTDOWN to proxy pipe: %ld", bytes_written);
        }
    }

    // Close PAL pipes
    if (PalHandleIsValid(g_gpu_proxy_pipe_enclave_to_proxy)) {
        PalObjectDestroy(&g_gpu_proxy_pipe_enclave_to_proxy);
        log_debug("Closed pipe: %s", GPU_PIPE_ENCLAVE_TO_PROXY);
    }
    if (PalHandleIsValid(g_gpu_proxy_pipe_proxy_to_enclave)) {
        PalObjectDestroy(&g_gpu_proxy_pipe_proxy_to_enclave);
        log_debug("Closed pipe: %s", GPU_PIPE_PROXY_TO_ENCLAVE);
    }

    // Unmap shared memory
    if (g_gpu_shared_mem) {
        // TODO: Implement PalDeviceUnmap or equivalent for the SHM region.
        // PalDeviceUnmap(g_gpu_shared_mem, g_gpu_shm_size);
        // If g_gpu_shared_mem was from malloc for testing, free it:
        // free(g_gpu_shared_mem);
        g_gpu_shared_mem = NULL;
        g_gpu_shm_size = 0;
        log_debug("Shared memory unmapped/freed (placeholder).");
    }

    // Clear session key and free GCM context
    if (g_session_key_initialized) {
        mbedtls_gcm_free(&g_aes_gcm_ctx);
        memset(g_gpu_session_key, 0, sizeof(g_gpu_session_key));
        g_session_key_initialized = false;
        log_debug("Session key cleared and GCM context freed.");
    }
    g_encryption_invocation_count = 0; // Reset IV counter
    return 0;
}


// Helper to construct IV for AES-GCM
// IV must be unique for each encryption with the same key.
// Using a 64-bit counter (padded to 12 bytes for GCM) should be safe.
static void construct_iv(uint64_t counter, uint8_t iv[12]) {
    memset(iv, 0, 12); // Zero out IV
    // Copy counter to the first 8 bytes of IV (little-endian).
    // mbedTLS GCM examples often use IV as a counter.
    memcpy(iv, &counter, sizeof(uint64_t));
    // The remaining 4 bytes are zero, or could be a fixed pattern if desired.
}

int gramine_cuda_malloc_device(gramine_device_ptr_t* device_ptr, size_t size) {
    log_debug("gramine_cuda_malloc_device called with size: %zu", size);
    if (!device_ptr) return -EINVAL;
    if (!PalHandleIsValid(g_gpu_proxy_pipe_enclave_to_proxy) || !g_session_key_initialized) {
        log_error("GPU IPC not initialized or session key missing.");
        return -EPIPE;
    }

    gpu_message_header_t req_header, resp_header;
    cmd_malloc_device_req_t malloc_req;
    cmd_malloc_device_resp_t malloc_resp;

    malloc_req.size = size;

    req_header.type = CMD_MALLOC_DEVICE;
    req_header.payload_size = sizeof(cmd_malloc_device_req_t);
    req_header.status = 0;

    int ret = send_command_receive_response(&req_header, &malloc_req, &resp_header, &malloc_resp, sizeof(cmd_malloc_device_resp_t));

    if (ret == 0 && resp_header.type == CMD_MALLOC_DEVICE && resp_header.status == 0) {
        // The proxy returns an opaque handle (e.g., an ID or index) that the enclave uses.
        // The proxy maps this opaque handle to the actual CUdeviceptr.
        *device_ptr = (gramine_device_ptr_t)malloc_resp.device_ptr_opaque;
        log_debug("CMD_MALLOC_DEVICE successful. Opaque device_ptr: %p", *device_ptr);
        return 0;
    } else {
        log_error("CMD_MALLOC_DEVICE failed. Proxy ret: %d, resp_status: %d, resp_type: %d", ret, resp_header.status, resp_header.type);
        return (ret != 0) ? ret : (resp_header.status != 0 ? resp_header.status : -1);
    }
}

int gramine_cuda_memcpy_host_to_device(gramine_device_ptr_t device_ptr, const void* host_ptr, size_t size) {
    log_debug("gramine_cuda_memcpy_host_to_device called with device_ptr: %p, host_ptr: %p, size: %zu",
                device_ptr, host_ptr, size);
    if (!PalHandleIsValid(g_gpu_proxy_pipe_enclave_to_proxy) || !g_session_key_initialized) {
        log_error("GPU IPC not initialized or session key missing.");
        return -EPIPE;
    }
    if (!g_gpu_shared_mem || g_gpu_shm_size < size) { // Ensure SHM is large enough for the current operation
        log_error("Shared memory not available or too small for H2D copy. SHM size: %zu, required: %zu", g_gpu_shm_size, size);
        return -ENOMEM;
    }

    // 1. Data to be copied is in `host_ptr`. Encrypt it into `g_gpu_shared_mem`.
    //    The entire SHM region is used as a temporary buffer for this single operation.
    if (size > g_gpu_shm_size) {
        log_error("H2D: requested size %zu exceeds SHM capacity %zu", size, g_gpu_shm_size);
        return -ENOMEM;
    }
    if (!g_gpu_shared_mem) { // Check if SHM pointer is valid
        log_error("H2D: Shared memory not initialized/mapped.");
        return -EFAULT;
    }

    g_encryption_invocation_count++;
    construct_iv(g_encryption_invocation_count, h2d_req.iv);

    // Encrypt host_ptr -> g_gpu_shared_mem (offset 0)
    int ret_crypto = mbedtls_gcm_crypt_and_tag(&g_aes_gcm_ctx, MBEDTLS_GCM_ENCRYPT, size,
                                               h2d_req.iv, sizeof(h2d_req.iv),
                                               NULL, 0, // No AAD
                                               (const unsigned char*)host_ptr, (unsigned char*)g_gpu_shared_mem,
                                               sizeof(h2d_req.tag), h2d_req.tag);
    if (ret_crypto != 0) {
        char error_buf[100];
        mbedtls_strerror(ret_crypto, error_buf, sizeof(error_buf));
        log_error("mbedtls_gcm_crypt_and_tag (encrypt) failed: %s (code: -0x%x)", error_buf, (unsigned int)-ret_crypto);
        return -ret_crypto;
    }
    log_debug("H2D: Encrypted %zu bytes to SHM. IV counter: %lu", size, g_encryption_invocation_count);

    h2d_req.device_ptr_opaque = (uint64_t)device_ptr;
    h2d_req.shm_offset = 0; // Using entire SHM as one buffer for now
    h2d_req.size = size;
    // IV and Tag are already set in h2d_req

    req_header.type = CMD_MEMCPY_H2D;
    req_header.payload_size = sizeof(cmd_memcpy_h2d_req_t);
    req_header.status = 0;

    int ret_ipc = send_command_receive_response(&req_header, &h2d_req, &resp_header, NULL, 0);

    if (ret_ipc == 0 && resp_header.type == CMD_MEMCPY_H2D && resp_header.status == 0) {
        log_debug("CMD_MEMCPY_H2D successful.");
        return 0;
    } else {
        log_error("CMD_MEMCPY_H2D failed. IPC ret: %d, resp_status: %d, resp_type: %d", ret_ipc, resp_header.status, resp_header.type);
        return (ret_ipc != 0) ? ret_ipc : (resp_header.status != 0 ? resp_header.status : -EIO);
    }
}

int gramine_cuda_memcpy_device_to_host(void* host_ptr, gramine_device_ptr_t device_ptr, size_t size) {
    log_debug("gramine_cuda_memcpy_device_to_host called with host_ptr: %p, device_ptr: %p, size: %zu",
                host_ptr, device_ptr, size);
    if (!PalHandleIsValid(g_gpu_proxy_pipe_enclave_to_proxy) || !g_session_key_initialized) {
        log_error("GPU IPC not initialized or session key missing.");
        return -EPIPE;
    }
    if (!g_gpu_shared_mem || g_gpu_shm_size < size) { // Ensure SHM is large enough
        log_error("Shared memory not available or too small for D2H copy. SHM size: %zu, required: %zu", g_gpu_shm_size, size);
        return -ENOMEM;
    }

    // uint64_t shm_offset = find_suitable_shm_offset_for_receiving_data(size);
    uint64_t shm_offset = 0; // Simplification: use offset 0. Needs proper management.

    gpu_message_header_t req_header, resp_header;
    cmd_memcpy_d2h_req_t d2h_req;

    d2h_req.device_ptr_opaque = (uint64_t)device_ptr;
    d2h_req.shm_offset = shm_offset;
    d2h_req.size = size;

    req_header.type = CMD_MEMCPY_D2H;
    req_header.payload_size = sizeof(cmd_memcpy_d2h_req_t);
    req_header.status = 0;

    int ret = send_command_receive_response(&req_header, &d2h_req, &resp_header, NULL, 0);

    if (ret == 0 && resp_header.type == CMD_MEMCPY_D2H && resp_header.status == 0) {
    // Proxy will place encrypted data in SHM. We need to decrypt it.
    cmd_memcpy_d2h_resp_t d2h_resp;

    int ret = send_command_receive_response(&req_header, &d2h_req, &resp_header, &d2h_resp, sizeof(cmd_memcpy_d2h_resp_t));

    if (ret == 0 && resp_header.type == CMD_MEMCPY_D2H && resp_header.status == 0) {
        if (d2h_resp.actual_size_written_to_shm > size) {
            log_error("D2H: Proxy reported writing %zu bytes, but max expected was %zu",
                      d2h_resp.actual_size_written_to_shm, size);
            return -EIO;
        }
        if (d2h_resp.actual_size_written_to_shm > g_gpu_shm_size) {
             log_error("D2H: Proxy reported writing %zu bytes, which exceeds SHM capacity %zu",
                      d2h_resp.actual_size_written_to_shm, g_gpu_shm_size);
            return -EIO;
        }

        // Decrypt data from g_gpu_shared_mem (offset 0) -> host_ptr
        ret = mbedtls_gcm_auth_decrypt(&g_aes_gcm_ctx, d2h_resp.actual_size_written_to_shm,
                                       d2h_resp.iv, sizeof(d2h_resp.iv),
                                       NULL, 0, // No AAD
                                       d2h_resp.tag, sizeof(d2h_resp.tag),
                                       (const unsigned char*)g_gpu_shared_mem, (unsigned char*)host_ptr);
        if (ret != 0) {
            char error_buf[100];
            mbedtls_strerror(ret, error_buf, sizeof(error_buf));
            log_error("mbedtls_gcm_auth_decrypt failed: %s", error_buf);
            return -ret;
        }
        log_debug("D2H: Decrypted %zu bytes from SHM.", d2h_resp.actual_size_written_to_shm);
        return 0;
    } else {
        log_error("CMD_MEMCPY_D2H failed. Proxy ret: %d, resp_status: %d, resp_type: %d", ret, resp_header.status, resp_header.type);
        return (ret != 0) ? ret : (resp_header.status != 0 ? resp_header.status : -EIO);
    }
}

int gramine_cuda_free_device(gramine_device_ptr_t device_ptr) {
    log_debug("gramine_cuda_free_device called with device_ptr: %p", device_ptr);
    if (!PalHandleIsValid(g_gpu_proxy_pipe_enclave_to_proxy) || !g_session_key_initialized) {
        log_error("GPU IPC not initialized or session key missing.");
        return -EPIPE;
    }

    gpu_message_header_t req_header, resp_header;
    cmd_free_device_req_t free_req;

    free_req.device_ptr_opaque = (uint64_t)device_ptr;

    req_header.type = CMD_FREE_DEVICE;
    req_header.payload_size = sizeof(cmd_free_device_req_t);
    req_header.status = 0;

    int ret = send_command_receive_response(&req_header, &free_req, &resp_header, NULL, 0);

    if (ret == 0 && resp_header.type == CMD_FREE_DEVICE && resp_header.status == 0) {
        log_debug("CMD_FREE_DEVICE successful.");
        return 0;
    } else {
        log_error("CMD_FREE_DEVICE failed. Proxy ret: %d, resp_status: %d, resp_type: %d", ret, resp_header.status, resp_header.type);
        return (ret != 0) ? ret : (resp_header.status != 0 ? resp_header.status : -1);
    }
}

int gramine_cuda_launch_kernel_by_name(const char* kernel_name, void** kernel_args,
                                       unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
                                       unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
                                       unsigned int shared_mem_bytes) {
    log_debug("gramine_cuda_launch_kernel_by_name called for kernel: %s", kernel_name);
    if (!PalHandleIsValid(g_gpu_proxy_pipe_enclave_to_proxy) || !g_session_key_initialized) {
        log_error("GPU IPC not initialized or session key missing.");
        return -EPIPE;
    }

    gpu_message_header_t req_header, resp_header;
    cmd_launch_kernel_req_t launch_req;

    strncpy(launch_req.kernel_name, kernel_name, MAX_KERNEL_NAME_LEN - 1);
    launch_req.kernel_name[MAX_KERNEL_NAME_LEN - 1] = '\0';

    // Simplified serialization for GEMM example: {A_gpu_opaque, B_gpu_opaque, C_gpu_opaque, N_val}
    // All are uint64_t for simplicity here, N_val is cast.
    // A more robust solution needs type information for each argument.
    if (kernel_args == NULL) {
        launch_req.serialized_args_len = 0;
    } else {
        // Assuming 4 arguments for GEMM: A, B, C (device_ptr_opaques), N (int, promoted to uint64_t for serialization)
        // This count should ideally be passed or derived from kernel_name.
        const int num_args_expected = 4;
        uint64_t* serialized_ptr = (uint64_t*)launch_req.serialized_args;
        int current_arg_idx = 0;

        for (int i = 0; i < num_args_expected && kernel_args[i] != NULL; ++i) {
            if ((current_arg_idx + 1) * sizeof(uint64_t) > MAX_SERIALIZED_KERNEL_ARGS_LEN) {
                log_error("Kernel args serialization exceeds buffer MAX_SERIALIZED_KERNEL_ARGS_LEN");
                return -ENOMEM;
            }
            // For device pointers (opaque handles), they are already gramine_device_ptr_t (uint64_t)
            // For int N, it's passed as int*. We dereference and cast to uint64_t for serialization.
            // This is specific to the GEMM example's argument structure.
            if (i < 3) { // A_gpu, B_gpu, C_gpu are gramine_device_ptr_t (which is void*, cast to uint64_t)
                serialized_ptr[current_arg_idx++] = (uint64_t)kernel_args[i];
            } else if (i == 3) { // N is int*
                int* n_val_ptr = (int*)kernel_args[i];
                serialized_ptr[current_arg_idx++] = (uint64_t)(*n_val_ptr);
            }
            // Add more types as needed, e.g., float, other scalars.
        }
        launch_req.serialized_args_len = current_arg_idx * sizeof(uint64_t);
        log_debug("Serialized %d kernel arguments, total size %u bytes.", current_arg_idx, launch_req.serialized_args_len);
    }


    launch_req.grid_dim_x = grid_dim_x;
    launch_req.grid_dim_y = grid_dim_y;
    launch_req.grid_dim_z = grid_dim_z;
    launch_req.block_dim_x = block_dim_x;
    launch_req.block_dim_y = block_dim_y;
    launch_req.block_dim_z = block_dim_z;
    launch_req.shared_mem_bytes = shared_mem_bytes;

    req_header.type = CMD_LAUNCH_KERNEL;
    req_header.payload_size = sizeof(cmd_launch_kernel_req_t);
    req_header.status = 0;

    int ret = send_command_receive_response(&req_header, &launch_req, &resp_header, NULL, 0);

    if (ret == 0 && resp_header.type == CMD_LAUNCH_KERNEL && resp_header.status == 0) {
        log_debug("CMD_LAUNCH_KERNEL successful for kernel: %s", kernel_name);
        return 0;
    } else {
        log_error("CMD_LAUNCH_KERNEL for %s failed. Proxy ret: %d, resp_status: %d, resp_type: %d",
                  kernel_name, ret, resp_header.status, resp_header.type);
        return (ret != 0) ? ret : (resp_header.status != 0 ? resp_header.status : -1);
    }
}
