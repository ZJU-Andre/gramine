#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h> // For sleep, close
#include <sys/stat.h> // For mkdir
#include <math.h>   // For fabs, if needed for float comparisons (not directly here but good for context)

// Gramine includes
#include "libos_ipc.h"
#include "libos_types.h"
#include "pal.h"
#include "libos_aes_gcm.h" // For AES-GCM operations

// Example-specific includes
#include "shared_service.h" // Common definitions for client/server (from ../common/)
#include "vector_add.h"     // For launch_vector_add_cuda
#include <onnxruntime_c_api.h> // For ONNX Runtime
#include <cublas_v2.h>      // For cuBLAS
#include <cuda_runtime.h>   // For cudaEvent_t

#define UNTRUSTED_SHM_PATH_PREFIX "/untrusted_region"
#define DEFAULT_ONNX_MODEL_PATH "/models/mobilenetv2-7.onnx"

static const unsigned char g_shared_enclave_aes_key[GCM_KEY_SIZE_BYTES] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f
};

// POC Session Key for Manifest Encryption/Decryption
static const unsigned char g_manifest_poc_session_key[GCM_KEY_SIZE_BYTES] = {
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f
};

static void generate_iv_for_response(unsigned char* iv, size_t iv_len) {
    for (size_t i = 0; i < iv_len; ++i) iv[i] = (unsigned char)(rand() % 256);
}

// --- Legacy Handlers (placeholders, unchanged from previous version) ---
int legacy_store_to_gpu(const char* id, const unsigned char* data, uint32_t size) { printf("SHARED_SERVICE_LOG: Legacy Storing to GPU (ID: %s, Size: %u) - Placeholder\n", id, size); if (!id || !data) return -EINVAL; return 0; }
int legacy_retrieve_from_gpu(const char* id, unsigned char* buffer, uint32_t buffer_max_size, uint32_t* size_read) { printf("SHARED_SERVICE_LOG: Legacy Retrieving from GPU (ID: %s) - Placeholder\n", id); if (!id || !buffer || !size_read) return -EINVAL; const char* dummy_data = "SampleLegacyGPUData"; uint32_t dummy_size = strlen(dummy_data); if (dummy_size > buffer_max_size) dummy_size = buffer_max_size; memcpy(buffer, dummy_data, dummy_size); *size_read = dummy_size; printf("SHARED_SERVICE_LOG: Retrieved %u bytes from GPU (ID: %s)\n", *size_read, id); return 0; }
void handle_legacy_store_request(const legacy_data_request_t* req, legacy_data_response_t* resp) { resp->data_size = 0;  if (req->sensitivity == SENSITIVITY_MEDIUM_GPU) { resp->status = legacy_store_to_gpu(req->path, req->data, req->data_size); } else if (req->sensitivity == SENSITIVITY_LOW_SHM) { char full_path[MAX_LEGACY_PATH_SIZE + sizeof(UNTRUSTED_SHM_PATH_PREFIX) + 1]; snprintf(full_path, sizeof(full_path), "%s/%s", UNTRUSTED_SHM_PATH_PREFIX, req->path); char* last_slash = strrchr(full_path, '/'); if (last_slash) { *last_slash = '\0';  if (mkdir(full_path, 0775) != 0 && errno != EEXIST) { perror("mkdir failed"); resp->status = -errno; *last_slash = '/'; return; } *last_slash = '/';  } FILE* fp = fopen(full_path, "wb"); if (!fp) { resp->status = -errno; return; } size_t written = fwrite(req->data, 1, req->data_size, fp); if (written != req->data_size) { resp->status = ferror(fp) ? -EIO : -ENOSPC; fclose(fp); return; } fclose(fp); resp->status = 0; } else { resp->status = -EINVAL; } }
void handle_legacy_retrieve_request(const legacy_data_request_t* req, legacy_data_response_t* resp) { if (req->sensitivity == SENSITIVITY_MEDIUM_GPU) { resp->status = legacy_retrieve_from_gpu(req->path, resp->data, sizeof(resp->data), &resp->data_size); } else if (req->sensitivity == SENSITIVITY_LOW_SHM) { char full_path[MAX_LEGACY_PATH_SIZE + sizeof(UNTRUSTED_SHM_PATH_PREFIX) + 1]; snprintf(full_path, sizeof(full_path), "%s/%s", UNTRUSTED_SHM_PATH_PREFIX, req->path); FILE* fp = fopen(full_path, "rb"); if (!fp) { resp->status = -errno; resp->data_size = 0; return; } size_t read_bytes = fread(resp->data, 1, sizeof(resp->data), fp); if (ferror(fp)) { resp->status = -EIO; resp->data_size = 0; fclose(fp); return; } fclose(fp); resp->data_size = read_bytes; resp->status = 0; } else { resp->status = -EINVAL; resp->data_size = 0; } }


// --- Vector Add Handler ---
static void handle_vector_add_request(const vector_add_request_payload_t* req, vector_add_response_payload_t* resp) {
    printf("SHARED_SERVICE_LOG: Handling VECTOR_ADD_REQUEST (masking: %s) for %u elements.\n",
           req->masking_level == MASKING_AES_GCM ? "AES-GCM" : "None",
           req->array_len_elements);

    resp->masking_level = req->masking_level; // Mirror request's masking level for response
    resp->array_len_elements = 0; // Default to 0 elements on error

    if (req->array_len_elements == 0 || req->array_len_elements > VECTOR_ARRAY_MAX_ELEMENTS) {
        resp->status = -EINVAL; return;
    }
    size_t data_sz = req->array_len_elements * sizeof(float); // Define data_sz early

    bool use_dma_for_output_a = (req->masking_level == MASKING_NONE && 
                                 req->dest_host_ptr_a != 0 && 
                                 req->dest_host_buffer_size_a >= data_sz);

    float *b_plain = NULL; // Will be allocated if not using DMA for B
    float *c_plain = NULL; // Will be allocated if not using DMA for C
    // a_result_plain is allocated as it's used as a fallback by launch_vector_add_cuda
    // and for AES-GCM output path.
    float *a_result_plain = (float*)malloc(data_sz); 

    if (!a_result_plain) { resp->status = -ENOMEM; goto cleanup_va; }

    bool use_dma_for_b = (req->masking_level == MASKING_NONE && req->src_device_ptr_b != 0);
    bool use_dma_for_c = (req->masking_level == MASKING_NONE && req->src_device_ptr_c != 0);

    if (use_dma_for_b) {
        printf("  DMA path for input B. Using client's device pointer: 0x%lx\n", req->src_device_ptr_b);
        // b_plain remains NULL, data is on client's GPU device memory
    } else {
        b_plain = (float*)malloc(data_sz);
        if (!b_plain) { resp->status = -ENOMEM; goto cleanup_va; }
        if (req->masking_level == MASKING_AES_GCM) {
            printf("  AES-GCM path for input B. Decrypting...\n");
            if (libos_aes_gcm_decrypt(g_shared_enclave_aes_key, req->iv_b, req->data_b, data_sz, req->tag_b, (unsigned char*)b_plain, NULL, 0) != 0) {
                fprintf(stderr, "SHARED_SERVICE_ERROR: Decryption failed for vector_add input B.\n");
                resp->status = -EPERM; goto cleanup_va; 
            }
        } else { // MASKING_NONE, but src_device_ptr_b is 0 (data in payload)
            printf("  Plaintext in payload path for input B. Copying...\n");
            memcpy(b_plain, req->data_b, data_sz);
        }
    }

    if (use_dma_for_c) {
        printf("  DMA path for input C. Using client's device pointer: 0x%lx\n", req->src_device_ptr_c);
        // c_plain remains NULL, data is on client's GPU device memory
    } else {
        c_plain = (float*)malloc(data_sz);
        if (!c_plain) { resp->status = -ENOMEM; goto cleanup_va; }
        if (req->masking_level == MASKING_AES_GCM) {
            printf("  AES-GCM path for input C. Decrypting...\n");
            if (libos_aes_gcm_decrypt(g_shared_enclave_aes_key, req->iv_c, req->data_c, data_sz, req->tag_c, (unsigned char*)c_plain, NULL, 0) != 0) {
                fprintf(stderr, "SHARED_SERVICE_ERROR: Decryption failed for vector_add input C.\n");
                resp->status = -EPERM; goto cleanup_va; 
            }
        } else { // MASKING_NONE, but src_device_ptr_c is 0 (data in payload)
            printf("  Plaintext in payload path for input C. Copying...\n");
            memcpy(c_plain, req->data_c, data_sz);
        }
    }
    
    int cuda_err_code = 0; const char* cuda_err_str = NULL;

    if (use_dma_for_output_a) {
        printf("  DMA path for output A. Client's host pointer: 0x%lx, buffer size: %u\n", 
               req->dest_host_ptr_a, req->dest_host_buffer_size_a);
        // TODO: CRITICAL SECURITY - Validate client-provided dest_host_ptr_a and dest_host_buffer_size_a.
        // Ensure the pointer is valid, within client's allowed memory, and size is sufficient.
        // This likely requires Gramine PAL support for safe cross-boundary memory access validation.
        // Current check (dest_host_buffer_size_a >= data_sz) is a basic sanity check.
    }

    // Updated call to launch_vector_add_cuda with output DMA parameters
    if (launch_vector_add_cuda(
            use_dma_for_output_a,       // Pass the flag for output DMA
            req->dest_host_ptr_a,       // Pass client's host pointer for output A
            a_result_plain,             // Pass host buffer as fallback
            b_plain,                    // Can be NULL if use_dma_for_b is true
            c_plain,                    // Can be NULL if use_dma_for_c is true
            req->array_len_elements, 
            &cuda_err_code, &cuda_err_str,
            use_dma_for_b, req->src_device_ptr_b,
            use_dma_for_c, req->src_device_ptr_c
        ) != 0) {
        fprintf(stderr, "SHARED_SERVICE_ERROR: launch_vector_add_cuda failed. Code: %d, Str: %s\n", cuda_err_code, cuda_err_str ? cuda_err_str : "N/A");
        resp->status = cuda_err_code != 0 ? cuda_err_code : -1; goto cleanup_va;
    }
    
    resp->array_len_elements = req->array_len_elements;

    if (use_dma_for_output_a) {
        printf("  Result vector A written directly to client's host memory via DMA.\n");
        // Data is directly in client's buffer. Zero out the payload part for data_a.
        // IV and Tag are not applicable for DMA output, ensure they are zeroed (already done by initial memset of resp).
        memset(resp->data_a, 0, sizeof(resp->data_a)); 
        // If resp was not fully memset at the start, zero IV and Tag here:
        // memset(resp->iv_a, 0, GCM_IV_SIZE_BYTES);
        // memset(resp->tag_a, 0, GCM_TAG_SIZE_BYTES);
    } else {
        // Fallback: Populate resp->data_a using a_result_plain (which launch_vector_add_cuda filled)
        if (resp->masking_level == MASKING_AES_GCM) {
            printf("  Encrypting result vector A (AES-GCM) from fallback buffer...\n");
            generate_iv_for_response(resp->iv_a, GCM_IV_SIZE_BYTES);
            if (libos_aes_gcm_encrypt(g_shared_enclave_aes_key, resp->iv_a, (unsigned char*)a_result_plain, data_sz, resp->data_a, resp->tag_a, NULL, 0) != 0) {
                fprintf(stderr, "SHARED_SERVICE_ERROR: Encryption failed for vector_add result.\n");
                resp->status = -EPERM; goto cleanup_va; 
            }
        } else { // MASKING_NONE (but not DMA output, so copy to payload)
            printf("  Copying plaintext result vector A from fallback buffer to response...\n");
            memcpy(resp->data_a, a_result_plain, data_sz);
        }
    }
    resp->status = 0;
cleanup_va:
    if (b_plain) free(b_plain); // Free only if allocated by service
    if (c_plain) free(c_plain); // Free only if allocated by service
    if (a_result_plain) free(a_result_plain);
    printf("SHARED_SERVICE_LOG: Finished VECTOR_ADD_REQUEST with status %d.\n", resp->status);
}

// --- ONNX Runtime Globals & Functions ---
static const OrtApi* g_ort_api = NULL; 
static OrtEnv* g_ort_env = NULL;
static OrtSession* g_ort_session = NULL;
static OrtAllocator* g_ort_allocator = NULL;
static const char* g_onnx_input_names[] = {"input"}; 
static const char* g_onnx_output_names[] = {"output"};

static int handle_ort_status(OrtStatus* status, const char* op_name) { /* ... (unchanged) ... */ 
    if (status) {
        const char* msg = g_ort_api ? g_ort_api->GetErrorMessage(status) : "ONNX API not available";
        fprintf(stderr, "SHARED_SERVICE_ONNX_ERROR: %s failed: %s\n", op_name, msg);
        if (g_ort_api) g_ort_api->ReleaseStatus(status);
        return -1; 
    }
    return 0; 
}
static int init_onnx_runtime(const char* model_path) { /* ... (unchanged) ... */ 
    printf("SHARED_SERVICE_LOG: Initializing ONNX Runtime (model: %s)...\n", model_path);
    g_ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort_api) { fprintf(stderr, "SHARED_SERVICE_ONNX_ERROR: Failed to get ONNX API base.\n"); return -1; }
    OrtStatus* ort_status = NULL;
    ort_status = g_ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "shared-service-onnx", &g_ort_env);
    if (handle_ort_status(ort_status, "CreateEnv") != 0) return -1;
    OrtSessionOptions* session_options;
    ort_status = g_ort_api->CreateSessionOptions(&session_options);
    if (handle_ort_status(ort_status, "CreateSessionOptions") != 0) { if(g_ort_env) g_ort_api->ReleaseEnv(g_ort_env); g_ort_env = NULL; return -1; }
    ort_status = g_ort_api->SessionOptionsAppendExecutionProvider_CUDA_V2(session_options, NULL);
    if (handle_ort_status(ort_status, "SessionOptionsAppendExecutionProvider_CUDA_V2") != 0) {
        fprintf(stderr, "SHARED_SERVICE_ONNX_WARNING: Failed to append CUDA EP. May fallback to CPU.\n");
    } else { printf("  CUDA Execution Provider configured.\n"); }
    ort_status = g_ort_api->CreateSession(g_ort_env, model_path, session_options, &g_ort_session);
    g_ort_api->ReleaseSessionOptions(session_options);
    if (handle_ort_status(ort_status, "CreateSession") != 0) { if(g_ort_env) g_ort_api->ReleaseEnv(g_ort_env); g_ort_env = NULL; return -1; }
    ort_status = g_ort_api->GetAllocatorWithDefaultOptions(&g_ort_allocator);
    if (handle_ort_status(ort_status, "GetAllocatorWithDefaultOptions") != 0) { 
        if(g_ort_session) g_ort_api->ReleaseSession(g_ort_session); g_ort_session = NULL;
        if(g_ort_env) g_ort_api->ReleaseEnv(g_ort_env); g_ort_env = NULL; return -1; 
    }
    printf("  ONNX Runtime initialized successfully.\n");
    return 0;
}
static void shutdown_onnx_runtime() { /* ... (unchanged) ... */ 
    printf("SHARED_SERVICE_LOG: Shutting down ONNX Runtime...\n");
    if (g_ort_api) { 
        if (g_ort_session) { g_ort_api->ReleaseSession(g_ort_session); g_ort_session = NULL; }
        if (g_ort_env) { g_ort_api->ReleaseEnv(g_ort_env); g_ort_env = NULL; }
        g_ort_allocator = NULL; g_ort_api = NULL; 
    }
    printf("SHARED_SERVICE_LOG: ONNX Runtime shut down.\n");
}

static void handle_onnx_inference_request(const onnx_inference_request_payload_t* req, onnx_inference_response_payload_t* resp) {
    printf("SHARED_SERVICE_LOG: Handling ONNX_INFERENCE_REQUEST (masking: %s, input size: %u bytes).\n",
           req->masking_level == MASKING_AES_GCM ? "AES-GCM" : "None",
           req->input_tensor_size_bytes);
    resp->status = -1; resp->output_tensor_size_bytes = 0;
    resp->masking_level = req->masking_level;

    if (!g_ort_api || !g_ort_session || !g_ort_allocator) { fprintf(stderr, "ONNX_ERROR: Runtime not initialized.\n"); return; }
    if (req->input_tensor_size_bytes == 0 || req->input_tensor_size_bytes > MAX_ONNX_INPUT_SIZE_BYTES) {
        resp->status = -EINVAL; return;
    }

    unsigned char* p_input_plaintext = NULL; // Host buffer for input, used if not DMA
    // d_enclave_input_tensor is no longer allocated by the enclave in the DMA path.
    // It will remain NULL if use_dma_input_tensor is true.
    float* d_enclave_input_tensor = NULL; 
    void* tensor_data_source = NULL; // Will point to p_input_plaintext or client's device_ptr

    bool use_dma_input_tensor = (req->masking_level == MASKING_NONE && req->src_device_ptr_input_tensor != 0);

    if (use_dma_input_tensor) {
        printf("  DMA path for ONNX input. Using client's device pointer directly: 0x%lx, Size: %u bytes\n",
               req->src_device_ptr_input_tensor, req->input_tensor_size_bytes);
        
        // Client's device pointer will be used directly. No allocation or DtoD copy in enclave.
        tensor_data_source = (void*)req->src_device_ptr_input_tensor;
        // d_enclave_input_tensor remains NULL for this path.

    } else { // Non-DMA path (AES-GCM or MASKING_NONE with data in payload)
        p_input_plaintext = (unsigned char*)malloc(req->input_tensor_size_bytes);
        if (!p_input_plaintext) { resp->status = -ENOMEM; goto cleanup_onnx; }

        if (req->masking_level == MASKING_AES_GCM) {
            printf("  AES-GCM path for ONNX input. Decrypting...\n");
            if (libos_aes_gcm_decrypt(g_shared_enclave_aes_key, req->iv, req->input_tensor, req->input_tensor_size_bytes, req->tag, p_input_plaintext, NULL, 0) != 0) {
                fprintf(stderr, "SHARED_SERVICE_ONNX_ERROR: Decryption of input tensor failed.\n");
                resp->status = -EPERM; goto cleanup_onnx;
            }
        } else { // MASKING_NONE, but src_device_ptr_input_tensor is 0 (data in payload)
            printf("  Plaintext in payload path for ONNX input. Copying...\n");
            memcpy(p_input_plaintext, req->input_tensor, req->input_tensor_size_bytes);
        }
        tensor_data_source = p_input_plaintext;
    }
    
    int64_t input_shape[] = {1, ONNX_MODEL_INPUT_CHANNELS, ONNX_MODEL_INPUT_HEIGHT, ONNX_MODEL_INPUT_WIDTH};
    // Ensure the input tensor size matches the expected model input dimensions
    if (req->input_tensor_size_bytes != (1 * ONNX_MODEL_INPUT_CHANNELS * ONNX_MODEL_INPUT_HEIGHT * ONNX_MODEL_INPUT_WIDTH * sizeof(float))) {
        fprintf(stderr, "SHARED_SERVICE_ONNX_ERROR: Input tensor size %u does not match expected model dimensions.\n", req->input_tensor_size_bytes);
        resp->status = -EINVAL; goto cleanup_onnx;
    }

    OrtValue *in_tensor = NULL, *out_tensor = NULL;
    OrtStatus* ort_status = NULL;
    
    // Create OrtValue pointing to the correct memory (client's device or enclave's host)
    // ONNX Runtime needs to know the memory type. For CUDA, it needs a OrtMemoryInfo.
    OrtMemoryInfo* mem_info = NULL;
    if (use_dma_input_tensor) { // Data is on client's GPU device memory
        printf("  Creating ONNX input tensor from client's device memory (0x%lx).\n", req->src_device_ptr_input_tensor);
        // Create OrtMemoryInfo for CUDA device memory
        // Assuming device_id 0. If multiple GPUs, this might need to be configurable or queried.
        ort_status = g_ort_api->CreateMemoryInfo("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDevice, &mem_info);
        if (handle_ort_status(ort_status, "CreateMemoryInfo_Cuda_DMA") != 0) goto cleanup_onnx;
    } else { // Data is on enclave's CPU host memory (in p_input_plaintext)
        printf("  Creating ONNX input tensor from enclave's host memory (%p).\n", p_input_plaintext);
        // OrtMemoryInfo for CPU memory (default)
        ort_status = g_ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info);
        if (handle_ort_status(ort_status, "CreateMemoryInfo_Cpu") != 0) goto cleanup_onnx;
    }

    // tensor_data_source now correctly points to either client's device_ptr or enclave's p_input_plaintext
    ort_status = g_ort_api->CreateTensorWithDataAsOrtValue(mem_info, tensor_data_source, req->input_tensor_size_bytes, input_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_tensor);
    if (g_ort_api && mem_info) g_ort_api->ReleaseMemoryInfo(mem_info); // Release mem_info after use
    if (handle_ort_status(ort_status, "CreateTensorWithDataAsOrtValue") != 0) goto cleanup_onnx;


    cudaEvent_t start_event, stop_event; float gpu_time_ms = 0.0f;
    cudaEventCreate(&start_event); cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, 0);
    ort_status = g_ort_api->Run(g_ort_session, NULL, g_onnx_input_names, (const OrtValue* const*)&in_tensor, 1, g_onnx_output_names, 1, &out_tensor);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event); 
    cudaEventElapsedTime(&gpu_time_ms, start_event, stop_event);
    cudaEventDestroy(start_event); cudaEventDestroy(stop_event);
    printf("SHARED_SERVICE_LOG: ONNX OrtRun (GPU execution) time: %.3f ms\n", gpu_time_ms);
    
    if (handle_ort_status(ort_status, "RunInference") != 0) goto cleanup_onnx;
    const float* out_data_ptr_const; // Use const pointer for GetTensorData
    ort_status = g_ort_api->GetTensorData(out_tensor, (const void**)&out_data_ptr_const);
    if (handle_ort_status(ort_status, "GetOutputData") != 0) goto cleanup_onnx;
    OrtTensorTypeAndShapeInfo* shape_info;
    ort_status = g_ort_api->GetTensorTypeAndShape(out_tensor, &shape_info);
    if (handle_ort_status(ort_status, "GetOutputShapeInfo") != 0) { if (shape_info) g_ort_api->ReleaseTensorTypeAndShapeInfo(shape_info); goto cleanup_onnx; }
    size_t num_out_elements;
    ort_status = g_ort_api->GetTensorShapeElementCount(shape_info, &num_out_elements);
    g_ort_api->ReleaseTensorTypeAndShapeInfo(shape_info);
    if (handle_ort_status(ort_status, "GetOutputElementCount") != 0) goto cleanup_onnx;
    uint32_t out_bytes = num_out_elements * sizeof(float);
    if (out_bytes > MAX_ONNX_OUTPUT_SIZE_BYTES) { 
        fprintf(stderr, "SHARED_SERVICE_ONNX_ERROR: Output tensor size %u exceeds MAX_ONNX_OUTPUT_SIZE_BYTES %d.\n", out_bytes, MAX_ONNX_OUTPUT_SIZE_BYTES);
        resp->status = -EFBIG; 
        goto cleanup_onnx; 
    }
    
    resp->output_tensor_size_bytes = out_bytes; // Set the actual size for the client

    bool use_dma_for_output = (req->masking_level == MASKING_NONE && 
                               req->dest_host_ptr_output != 0);

    if (use_dma_for_output) {
        if (req->dest_host_buffer_size_output >= out_bytes) {
            printf("  DMA path for ONNX output. Client host pointer: 0x%lx, buffer size: %u, data size: %u\n",
                   req->dest_host_ptr_output, req->dest_host_buffer_size_output, out_bytes);

            // TODO: CRITICAL SECURITY - Validate client-provided dest_host_ptr_output and dest_host_buffer_size_output.
            // Ensure the pointer is valid, within client's allowed memory, and size is sufficient.
            // This likely requires Gramine PAL support. For now, we rely on the client providing a valid pinned memory pointer.

            cudaError_t cpy_err = cudaMemcpy((void*)req->dest_host_ptr_output, 
                                             out_data_ptr_const, // This is assumed to be a device pointer from ORT Run
                                             out_bytes, 
                                             cudaMemcpyDeviceToHost);
            if (cpy_err != cudaSuccess) {
                fprintf(stderr, "SHARED_SERVICE_ONNX_ERROR: cudaMemcpy DtoH to client's host pointer failed: %s\n", cudaGetErrorString(cpy_err));
                resp->status = -EIO; // Indicate I/O error
                // Fallback to payload path is complex here as client expects DMA or error. 
                // For now, just error out.
                goto cleanup_onnx;
            }
            printf("  ONNX output tensor (%u bytes) copied directly to client's host memory via DMA.\n", out_bytes);
            // IV and Tag are not used for DMA output, ensure they are zeroed (should be by initial memset of resp)
            memset(resp->output_tensor, 0, sizeof(resp->output_tensor)); 
            memset(resp->iv, 0, sizeof(resp->iv));
            memset(resp->tag, 0, sizeof(resp->tag));
        } else {
            fprintf(stderr, "SHARED_SERVICE_ONNX_WARNING: DMA for output requested, but client buffer size (%u) is too small for output data (%u). Falling back to payload.\n",
                    req->dest_host_buffer_size_output, out_bytes);
            use_dma_for_output = false; // Disable DMA output and fallback
        }
    }

    if (!use_dma_for_output) { // Fallback or non-DMA path
        if (resp->masking_level == MASKING_AES_GCM) {
            printf("  Encrypting output tensor (AES-GCM) into response payload...\n");
            generate_iv_for_response(resp->iv, GCM_IV_SIZE_BYTES);
            if (libos_aes_gcm_encrypt(g_shared_enclave_aes_key, resp->iv, (unsigned char*)out_data_ptr_const, out_bytes, resp->output_tensor, resp->tag, NULL, 0) != 0) {
                fprintf(stderr, "SHARED_SERVICE_ONNX_ERROR: Encryption of output tensor failed.\n");
                resp->status = -EPERM; goto cleanup_onnx;
            }
        } else { // MASKING_NONE (and not DMA, or DMA fallback)
            printf("  Copying plaintext output tensor to response payload...\n");
            memcpy(resp->output_tensor, out_data_ptr_const, out_bytes);
        }
    }
    resp->status = 0;
cleanup_onnx:
    if (g_ort_api) { 
        if (in_tensor) g_ort_api->ReleaseValue(in_tensor); 
        if (out_tensor) g_ort_api->ReleaseValue(out_tensor); 
    }
    if (p_input_plaintext) { // Free host buffer if it was allocated (non-DMA path)
        printf("  Freeing host buffer p_input_plaintext.\n");
        free(p_input_plaintext);
        p_input_plaintext = NULL;
    }
    // d_enclave_input_tensor is only allocated in the non-DMA path if that path were to use
    // a separate device buffer. Currently, the non-DMA path uses host memory (p_input_plaintext)
    // and CreateTensorWithDataAsOrtValue handles the HtoD copy internally if needed by ORT.
    // If d_enclave_input_tensor was allocated (e.g., in a previous version or a different logic branch),
    // this would free it. In the current DMA path, d_enclave_input_tensor is NULL.
    if (d_enclave_input_tensor) { 
        printf("  Freeing enclave's device buffer d_enclave_input_tensor (%p) - this should not happen in DMA path.\n", d_enclave_input_tensor);
        cudaError_t free_err = cudaFree(d_enclave_input_tensor);
        if (free_err != cudaSuccess) {
            fprintf(stderr, "SHARED_SERVICE_ONNX_ERROR: cudaFree for d_enclave_input_tensor failed: %s\n", cudaGetErrorString(free_err));
            if (resp->status == 0) resp->status = -EIO;
        }
        d_enclave_input_tensor = NULL;
    }
    printf("SHARED_SERVICE_LOG: Finished ONNX_INFERENCE_REQUEST with status %d.\n", resp->status);
}

// --- cuBLAS Globals & Functions ---
static cublasHandle_t g_cublas_handle = NULL;

static int init_cublas() { /* ... (unchanged) ... */ 
    printf("SHARED_SERVICE_LOG: Initializing cuBLAS...\n");
    cublasStatus_t status = cublasCreate(&g_cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "SHARED_SERVICE_CUBLAS_ERROR: cublasCreate failed: %s\n", cublasGetErrorString(status));
        return -1;
    }
    printf("SHARED_SERVICE_LOG: cuBLAS initialized successfully.\n");
    return 0;
}
static void shutdown_cublas() { /* ... (unchanged) ... */ 
    printf("SHARED_SERVICE_LOG: Shutting down cuBLAS...\n");
    if (g_cublas_handle) {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = NULL;
    }
    printf("SHARED_SERVICE_LOG: cuBLAS shut down.\n");
}

static void handle_gemm_request(const gemm_request_payload_t* req, gemm_response_payload_t* resp) {
    printf("SHARED_SERVICE_LOG: Handling GEMM_REQUEST (masking: %s, M=%d, N=%d, K=%d)\n",
           req->masking_level == MASKING_AES_GCM ? "AES-GCM" : "None",
           req->M, req->N, req->K);
    resp->status = -1; resp->matrix_c_size_bytes = 0;
    resp->masking_level = req->masking_level;

    if (!g_cublas_handle) { fprintf(stderr, "CUBLAS_ERROR: cuBLAS not initialized.\n"); return; }
    if (req->M <= 0 || req->N <= 0 || req->K <= 0 ||
        req->M > MAX_GEMM_DIM_SIZE || req->N > MAX_GEMM_DIM_SIZE || req->K > MAX_GEMM_DIM_SIZE ||
        req->matrix_a_size_bytes != ( (size_t)req->M * req->K * sizeof(float) ) ||
        req->matrix_b_size_bytes != ( (size_t)req->K * req->N * sizeof(float) ) ) {
        resp->status = -EINVAL; return;
    }

    float *pA_plain = NULL; // Host buffer for matrix A, used if not DMA
    float *pB_plain = NULL; // Host buffer for matrix B, used if not DMA
    float *pC_result_plain = NULL; // Host buffer for result matrix C (always allocated)

    float *dA_enclave_allocated = NULL; // Device buffer for A if allocated by enclave
    float *dB_enclave_allocated = NULL; // Device buffer for B if allocated by enclave
    float *dC = NULL;                   // Device buffer for C (always allocated by enclave)

    float *dA_effective = NULL; // Effective device pointer for A used in cuBLAS call
    float *dB_effective = NULL; // Effective device pointer for B used in cuBLAS call
    
    cudaEvent_t start_event, stop_event; float gpu_time_ms = 0.0f;
    cudaError_t cuda_err; // Used for CUDA calls not directly assigning to resp->status

    bool use_dma_matrix_a = (req->masking_level == MASKING_NONE && req->src_device_ptr_matrix_a != 0);
    bool use_dma_matrix_b = (req->masking_level == MASKING_NONE && req->src_device_ptr_matrix_b != 0);
    
    resp->matrix_c_size_bytes = (size_t)req->M * req->N * sizeof(float); // Calculate size of C
    if (resp->matrix_c_size_bytes > MAX_GEMM_MATRIX_SIZE_BYTES) { 
        fprintf(stderr, "SHARED_SERVICE_CUBLAS_ERROR: Result matrix C size %zu exceeds MAX_GEMM_MATRIX_SIZE_BYTES %d.\n",
                resp->matrix_c_size_bytes, MAX_GEMM_MATRIX_SIZE_BYTES);
        resp->status = -EFBIG; goto cleanup_gemm; 
    }

    bool use_dma_for_output_c = (req->masking_level == MASKING_NONE && 
                                 req->dest_host_ptr_c != 0 &&
                                 req->dest_host_buffer_size_c >= resp->matrix_c_size_bytes);

    // pC_result_plain is allocated as it's used as a fallback or for AES-GCM path
    pC_result_plain = (float*)malloc(resp->matrix_c_size_bytes);
    if (!pC_result_plain) { 
        fprintf(stderr, "SHARED_SERVICE_CUBLAS_ERROR: Failed to allocate host memory for pC_result_plain.\n");
        resp->status = -ENOMEM; goto cleanup_gemm; 
    }

    // Handle Matrix A
    if (use_dma_matrix_a) {
        printf("  DMA path for GEMM Matrix A. Using client's device pointer: 0x%lx\n", req->src_device_ptr_matrix_a);
        dA_effective = (float*)req->src_device_ptr_matrix_a; // Use client's device pointer directly
    } else {
        pA_plain = (float*)malloc(req->matrix_a_size_bytes);
        if (!pA_plain) { resp->status = -ENOMEM; goto cleanup_gemm; }

        if (req->masking_level == MASKING_AES_GCM) {
            printf("  AES-GCM path for GEMM Matrix A. Decrypting...\n");
            if (libos_aes_gcm_decrypt(g_shared_enclave_aes_key, req->iv_a, req->matrix_a, req->matrix_a_size_bytes, req->tag_a, (unsigned char*)pA_plain, NULL, 0) != 0) {
                fprintf(stderr, "SHARED_SERVICE_CUBLAS_ERROR: Decryption failed for GEMM Matrix A.\n");
                resp->status = -EPERM; goto cleanup_gemm;
            }
        } else { // MASKING_NONE, but src_device_ptr_matrix_a is 0 (data in payload)
            printf("  Plaintext in payload path for GEMM Matrix A. Copying from payload...\n");
            memcpy(pA_plain, req->matrix_a, req->matrix_a_size_bytes);
        }
        
        cuda_err = cudaMalloc((void**)&dA_enclave_allocated, req->matrix_a_size_bytes);
        if (cuda_err != cudaSuccess) { 
            fprintf(stderr, "SHARED_SERVICE_CUBLAS_ERROR: cudaMalloc for dA_enclave_allocated failed: %s\n", cudaGetErrorString(cuda_err));
            resp->status = -ENOMEM; goto cleanup_gemm_gpu; 
        }
        cuda_err = cudaMemcpy(dA_enclave_allocated, pA_plain, req->matrix_a_size_bytes, cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) { 
            fprintf(stderr, "SHARED_SERVICE_CUBLAS_ERROR: cudaMemcpy HtoD for dA_enclave_allocated failed: %s\n", cudaGetErrorString(cuda_err));
            resp->status = -EIO; goto cleanup_gemm_gpu; 
        }
        dA_effective = dA_enclave_allocated;
    }

    // Handle Matrix B
    if (use_dma_matrix_b) {
        printf("  DMA path for GEMM Matrix B. Using client's device pointer: 0x%lx\n", req->src_device_ptr_matrix_b);
        dB_effective = (float*)req->src_device_ptr_matrix_b; // Use client's device pointer directly
    } else {
        pB_plain = (float*)malloc(req->matrix_b_size_bytes);
        if (!pB_plain) { resp->status = -ENOMEM; goto cleanup_gemm; }

        if (req->masking_level == MASKING_AES_GCM) {
            printf("  AES-GCM path for GEMM Matrix B. Decrypting...\n");
            if (libos_aes_gcm_decrypt(g_shared_enclave_aes_key, req->iv_b, req->matrix_b, req->matrix_b_size_bytes, req->tag_b, (unsigned char*)pB_plain, NULL, 0) != 0) {
                fprintf(stderr, "SHARED_SERVICE_CUBLAS_ERROR: Decryption failed for GEMM Matrix B.\n");
                resp->status = -EPERM; goto cleanup_gemm;
            }
        } else { // MASKING_NONE, but src_device_ptr_matrix_b is 0
            printf("  Plaintext in payload path for GEMM Matrix B. Copying from payload...\n");
            memcpy(pB_plain, req->matrix_b, req->matrix_b_size_bytes);
        }

        cuda_err = cudaMalloc((void**)&dB_enclave_allocated, req->matrix_b_size_bytes);
        if (cuda_err != cudaSuccess) { 
            fprintf(stderr, "SHARED_SERVICE_CUBLAS_ERROR: cudaMalloc for dB_enclave_allocated failed: %s\n", cudaGetErrorString(cuda_err));
            resp->status = -ENOMEM; goto cleanup_gemm_gpu; 
        }
        cuda_err = cudaMemcpy(dB_enclave_allocated, pB_plain, req->matrix_b_size_bytes, cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) { 
            fprintf(stderr, "SHARED_SERVICE_CUBLAS_ERROR: cudaMemcpy HtoD for dB_enclave_allocated failed: %s\n", cudaGetErrorString(cuda_err));
            resp->status = -EIO; goto cleanup_gemm_gpu; 
        }
        dB_effective = dB_enclave_allocated;
    }

    // Allocate device memory for result matrix C (always done by enclave)
    cuda_err = cudaMalloc((void**)&dC, resp->matrix_c_size_bytes);
    if (cuda_err != cudaSuccess) { 
        fprintf(stderr, "SHARED_SERVICE_CUBLAS_ERROR: cudaMalloc for dC failed: %s\n", cudaGetErrorString(cuda_err));
        resp->status = -ENOMEM; goto cleanup_gemm_gpu; 
    }

    const float alpha = 1.0f, beta = 0.0f;
    cudaEventCreate(&start_event); cudaEventCreate(&stop_event);
    cudaEventRecord(start_event, 0);
    // Use effective device pointers for A and B in cuBLAS call
    cublasStatus_t cblas_stat = cublasSgemm(g_cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T,
                                           req->N, req->M, req->K, &alpha,
                                           dB_effective, req->K, dA_effective, req->M, &beta, dC, req->N);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&gpu_time_ms, start_event, stop_event);
    cudaEventDestroy(start_event); cudaEventDestroy(stop_event);
    printf("SHARED_SERVICE_LOG: cuBLAS SGEMM (GPU execution) time: %.3f ms\n", gpu_time_ms);

    if (cblas_stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "SHARED_SERVICE_CUBLAS_ERROR: cublasSgemm failed: %s\n", cublasGetErrorString(cblas_stat));
        resp->status = -EIO; goto cleanup_gemm_gpu;
    }

    if (use_dma_for_output_c) {
        printf("  DMA path for output Matrix C. Client's host pointer: 0x%lx, buffer size: %u, data size: %u\n",
               req->dest_host_ptr_c, req->dest_host_buffer_size_c, resp->matrix_c_size_bytes);
        
        // TODO: CRITICAL SECURITY - Validate client-provided dest_host_ptr_c and dest_host_buffer_size_c.
        // This is vital to prevent writes to arbitrary enclave/host memory.
        // Requires robust Gramine PAL support for cross-boundary memory validation.

        cuda_err = cudaMemcpy((void*)req->dest_host_ptr_c, dC, resp->matrix_c_size_bytes, cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "SHARED_SERVICE_CUBLAS_ERROR: cudaMemcpy DtoH to client's host pointer failed: %s\n", cudaGetErrorString(cuda_err));
            resp->status = -EIO;
            goto cleanup_gemm_gpu; // Error out if DMA copy fails
        }
        printf("  Matrix C (%u bytes) copied directly to client's host memory via DMA.\n", resp->matrix_c_size_bytes);
        // Zero out the payload part for matrix_c, iv_c, tag_c as they are not used.
        memset(resp->matrix_c, 0, sizeof(resp->matrix_c));
        memset(resp->iv_c, 0, sizeof(resp->iv_c));
        memset(resp->tag_c, 0, sizeof(resp->tag_c));
    } else {
        // Fallback: Copy dC to pC_result_plain first
        cuda_err = cudaMemcpy(pC_result_plain, dC, resp->matrix_c_size_bytes, cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "SHARED_SERVICE_CUBLAS_ERROR: cudaMemcpy DtoH to pC_result_plain failed: %s\n", cudaGetErrorString(cuda_err));
            resp->status = -EIO; goto cleanup_gemm_gpu;
        }

        // Then handle AES-GCM or plaintext copy to response payload from pC_result_plain
        if (resp->masking_level == MASKING_AES_GCM) {
            printf("  Encrypting result matrix C (AES-GCM) from fallback buffer into response payload...\n");
            generate_iv_for_response(resp->iv_c, GCM_IV_SIZE_BYTES);
            if (libos_aes_gcm_encrypt(g_shared_enclave_aes_key, resp->iv_c, (unsigned char*)pC_result_plain, resp->matrix_c_size_bytes, resp->matrix_c, resp->tag_c, NULL, 0) != 0) {
                fprintf(stderr, "SHARED_SERVICE_CUBLAS_ERROR: Encryption failed for GEMM result.\n");
                resp->status = -EPERM; goto cleanup_gemm_gpu;
            }
        } else { // MASKING_NONE (but not DMA output, or DMA fallback due to size mismatch)
            printf("  Copying plaintext result matrix C from fallback buffer to response payload...\n");
            memcpy(resp->matrix_c, pC_result_plain, resp->matrix_c_size_bytes);
        }
    }
    resp->status = 0;

cleanup_gemm_gpu:
    // Free device memory: dC is always enclave-allocated.
    // dA_enclave_allocated and dB_enclave_allocated are freed only if they were allocated by enclave.
    if (dC) { cudaFree(dC); dC = NULL; }
    if (dA_enclave_allocated) { cudaFree(dA_enclave_allocated); dA_enclave_allocated = NULL; }
    if (dB_enclave_allocated) { cudaFree(dB_enclave_allocated); dB_enclave_allocated = NULL; }
    // Note: dA_effective and dB_effective are just pointers, not separate allocations.
    // If they pointed to client memory, client is responsible for that.
    
cleanup_gemm:
    // Free host memory: pC_result_plain is always enclave-allocated.
    // pA_plain and pB_plain are freed only if they were allocated by enclave.
    if (pC_result_plain) { free(pC_result_plain); pC_result_plain = NULL; }
    if (pA_plain) { free(pA_plain); pA_plain = NULL; }
    if (pB_plain) { free(pB_plain); pB_plain = NULL; }
    
    printf("SHARED_SERVICE_LOG: Finished GEMM_REQUEST with status %d.\n", resp->status);
}

// --- Batch GPU Request Handler ---
static void handle_batch_gpu_request(const batch_gpu_request_payload_t* req, batch_gpu_response_payload_t* resp) {
    printf("SHARED_SERVICE_LOG: Handling BATCH_GPU_REQUEST...\n");
    resp->overall_batch_status = -1; // Default to error
    resp->num_segments = 0;      // Initialize to 0, will be set later based on processed outputs

    workload_manifest_t plaintext_manifest;
    memset(&plaintext_manifest, 0, sizeof(workload_manifest_t));

    // 1. Decrypt the manifest
    printf("  Decrypting workload manifest (size: %u bytes)...\n", req->encrypted_manifest_size);
    if (req->encrypted_manifest_size == 0 || req->encrypted_manifest_size > sizeof(workload_manifest_t)) {
        fprintf(stderr, "SHARED_SERVICE_BATCH_ERROR: Invalid encrypted manifest size: %u.\n", req->encrypted_manifest_size);
        resp->overall_batch_status = -EINVAL;
        return;
    }

    int dec_ret = libos_aes_gcm_decrypt(g_manifest_poc_session_key, 
                                        req->encrypted_manifest_iv,
                                        req->encrypted_manifest_data,
                                        req->encrypted_manifest_size,
                                        req->encrypted_manifest_tag,
                                        (unsigned char*)&plaintext_manifest, // Output buffer
                                        NULL, 0); // No AAD

    if (dec_ret != 0) {
        fprintf(stderr, "SHARED_SERVICE_BATCH_ERROR: Manifest decryption failed. Status: %d\n", dec_ret);
        resp->overall_batch_status = -EPERM; // Permission denied or decryption error
        return;
    }
    printf("  Manifest decrypted successfully.\n");

    // 2. Basic Manifest Parsing/Validation
    if (plaintext_manifest.num_segments == 0 || plaintext_manifest.num_segments > MAX_SEGMENTS_PER_MANIFEST) {
        fprintf(stderr, "SHARED_SERVICE_BATCH_ERROR: Invalid num_segments (%u) in decrypted manifest. Max allowed: %d.\n",
                plaintext_manifest.num_segments, MAX_SEGMENTS_PER_MANIFEST);
        resp->overall_batch_status = -EINVAL;
        return;
    }
    // Ensure manifest_id is null-terminated before logging, as it's a char array
    char manifest_id_for_log[MAX_MANIFEST_ID_LEN + 1];
    strncpy(manifest_id_for_log, plaintext_manifest.manifest_id, MAX_MANIFEST_ID_LEN);
    manifest_id_for_log[MAX_MANIFEST_ID_LEN] = '\0';

    printf("  Parsed Manifest ID: '%s', Number of Segments: %u\n", 
           manifest_id_for_log, plaintext_manifest.num_segments);

    // TODO: Placeholder for actual segment processing logic using CUDA streams, etc.
    
    // --- POC: Input Staging Logic ---
    typedef struct {
        char segment_id[MAX_SEGMENT_ID_LEN]; // From manifest
        gpu_data_masking_level_t masking_level; // From IPC descriptor
        int direction; // From manifest (0:input, 1:output, 2:input/output)
        char gpu_operation_id[MAX_GPU_OPERATION_ID_LEN]; // From manifest

        // For staged input data
        unsigned char* enclave_host_buffer_input; 
        void* enclave_device_buffer_input;    
        void* effective_gpu_input_ptr;        
        uint32_t processed_input_data_size;

        // For staging output data (to be used in later subtask)
        void* enclave_device_buffer_output;   
        unsigned char* enclave_host_buffer_output; 

        int status; // 0 for success
    } SegmentRuntimeInfoPoc;

    SegmentRuntimeInfoPoc segment_run_info[MAX_SEGMENTS_PER_MANIFEST]; // MAX_SEGMENTS_PER_MANIFEST from shared_service.h
    memset(segment_run_info, 0, sizeof(segment_run_info));

    bool any_segment_failed = false;

    for (uint32_t i = 0; i < decrypted_manifest.num_segments; ++i) {
        workload_manifest_segment_t* manifest_seg = &decrypted_manifest.segments[i];
        ipc_segment_descriptor_t* ipc_seg = &req->segment_descriptors[i]; // Assuming order matches
        SegmentRuntimeInfoPoc* run_seg = &segment_run_info[i];

        strncpy(run_seg->segment_id, manifest_seg->segment_id, MAX_SEGMENT_ID_LEN -1);
        run_seg->segment_id[MAX_SEGMENT_ID_LEN-1] = '\0';
        strncpy(run_seg->gpu_operation_id, manifest_seg->gpu_operation_id, MAX_GPU_OPERATION_ID_LEN -1);
        run_seg->gpu_operation_id[MAX_GPU_OP_ID_LEN-1] = '\0';
        run_seg->direction = manifest_seg->direction;
        run_seg->masking_level = ipc_seg->masking_level; // Masking level from IPC descriptor
        run_seg->status = 0; // Default to success for this segment

        if (run_seg->direction == DATA_DIRECTION_INPUT || run_seg->direction == DATA_DIRECTION_INPUT_OUTPUT) {
            printf("  Staging input for segment ID: %s, GPU Op ID: %s\n", run_seg->segment_id, run_seg->gpu_operation_id);
            if (run_seg->masking_level == MASKING_AES_GCM) {
                printf("    Staging AES-GCM input for segment %s...\n", run_seg->segment_id);
                if (ipc_seg->encrypted_data_size == 0 || ipc_seg->encrypted_data_size > MAX_INLINE_SEGMENT_DATA_SIZE) {
                    fprintf(stderr, "    ERROR: Invalid encrypted_data_size %u for segment %s.\n", ipc_seg->encrypted_data_size, run_seg->segment_id);
                    run_seg->status = -EINVAL; any_segment_failed = true; continue;
                }

                run_seg->enclave_host_buffer_input = (unsigned char*)malloc(ipc_seg->encrypted_data_size);
                if (!run_seg->enclave_host_buffer_input) {
                    fprintf(stderr, "    ERROR: Failed to malloc enclave_host_buffer_input for segment %s.\n", run_seg->segment_id);
                    run_seg->status = -ENOMEM; any_segment_failed = true; continue;
                }

                if (libos_aes_gcm_decrypt(g_shared_enclave_aes_key, ipc_seg->iv, ipc_seg->encrypted_data, 
                                          ipc_seg->encrypted_data_size, ipc_seg->tag, 
                                          run_seg->enclave_host_buffer_input, NULL, 0) != 0) {
                    fprintf(stderr, "    ERROR: Decryption failed for segment %s.\n", run_seg->segment_id);
                    run_seg->status = -EPERM; any_segment_failed = true; free(run_seg->enclave_host_buffer_input); run_seg->enclave_host_buffer_input = NULL; continue;
                }
                
                cudaError_t cuda_err = cudaMalloc(&run_seg->enclave_device_buffer_input, ipc_seg->encrypted_data_size);
                if (cuda_err != cudaSuccess) {
                    fprintf(stderr, "    ERROR: cudaMalloc for enclave_device_buffer_input failed for segment %s: %s\n", run_seg->segment_id, cudaGetErrorString(cuda_err));
                    run_seg->status = -ENOMEM; any_segment_failed = true; free(run_seg->enclave_host_buffer_input); run_seg->enclave_host_buffer_input = NULL; continue;
                }

                cuda_err = cudaMemcpy(run_seg->enclave_device_buffer_input, run_seg->enclave_host_buffer_input, ipc_seg->encrypted_data_size, cudaMemcpyHostToDevice);
                if (cuda_err != cudaSuccess) {
                    fprintf(stderr, "    ERROR: cudaMemcpy HtoD failed for segment %s: %s\n", run_seg->segment_id, cudaGetErrorString(cuda_err));
                    run_seg->status = -EIO; any_segment_failed = true; cudaFree(run_seg->enclave_device_buffer_input); run_seg->enclave_device_buffer_input = NULL; free(run_seg->enclave_host_buffer_input); run_seg->enclave_host_buffer_input = NULL; continue;
                }
                run_seg->effective_gpu_input_ptr = run_seg->enclave_device_buffer_input;
                run_seg->processed_input_data_size = ipc_seg->encrypted_data_size; // Plaintext size is same as ciphertext for AES-GCM
                printf("    AES-GCM Input segment %s staged successfully to device memory %p, size %u.\n", run_seg->segment_id, run_seg->effective_gpu_input_ptr, run_seg->processed_input_data_size);

            } else if (run_seg->masking_level == MASKING_NONE) { // DMA input path
                printf("    Staging MASKING_NONE input for segment %s...\n", run_seg->segment_id);
                run_seg->processed_input_data_size = ipc_seg->input_data_size;
                if (ipc_seg->src_device_ptr == 0) {
                    fprintf(stderr, "    ERROR: src_device_ptr is NULL for DMA segment %s.\n", run_seg->segment_id);
                    run_seg->status = -EINVAL; any_segment_failed = true; continue;
                }
                // TODO: SECURITY - Implement robust DMA input pointer validation.
                // This is a placeholder for where more advanced checks would go.
                // For POC, we're trusting the client's pointer if it's not NULL.
                run_seg->effective_gpu_input_ptr = (void*)ipc_seg->src_device_ptr;
                printf("    MASKING_NONE Input segment %s staged (using client device_ptr %p, size %u).\n", run_seg->segment_id, run_seg->effective_gpu_input_ptr, run_seg->processed_input_data_size);
            }
        }
    }
    printf("  SHARED_SERVICE_BATCH_LOG: Input staging phase complete.\n");
    printf("  SHARED_SERVICE_BATCH_LOG: TODO: Implement GPU operation dispatch using CUDA streams based on gpu_operation_id.\n");
    printf("  SHARED_SERVICE_BATCH_LOG: TODO: Implement output finalization (encryption/DMA setup for output segments).\n");
    printf("  SHARED_SERVICE_BATCH_LOG: TODO: Implement robust error handling and resource management per segment.\n");

    // Populate dummy response based on new segment_run_info
    resp->num_segments = decrypted_manifest.num_segments; 
    for (uint32_t i = 0; i < decrypted_manifest.num_segments; ++i) {
        strncpy(resp->segments[i].segment_id, segment_run_info[i].segment_id, MAX_SEGMENT_ID_LEN -1);
        resp->segments[i].segment_id[MAX_SEGMENT_ID_LEN-1] = '\0';
        resp->segments[i].status = segment_run_info[i].status; // Reflect staging status for now
        
        // For output segments, set dummy actual_output_data_size for POC
        if (segment_run_info[i].direction == DATA_DIRECTION_OUTPUT || segment_run_info[i].direction == DATA_DIRECTION_INPUT_OUTPUT) {
             if (segment_run_info[i].masking_level == MASKING_AES_GCM) {
                resp->segments[i].actual_output_data_size = MAX_INLINE_SEGMENT_DATA_SIZE / 2; // Dummy
                resp->segments[i].encrypted_data_size = resp->segments[i].actual_output_data_size;
                generate_iv_for_response(resp->segments[i].iv, GCM_IV_SIZE_BYTES);
                memset(resp->segments[i].tag, 0xBC, GCM_TAG_SIZE_BYTES); 
             } else { // MASKING_NONE
                resp->segments[i].actual_output_data_size = decrypted_manifest.segments[i].client_dma_output_buffer_size > 0 ? decrypted_manifest.segments[i].client_dma_output_buffer_size : 32;
                resp->segments[i].encrypted_data_size = 0; // No data in payload for DMA output
             }
        } else {
            resp->segments[i].actual_output_data_size = 0;
            resp->segments[i].encrypted_data_size = 0;
        }
    }

    if (any_segment_failed) {
        resp->overall_batch_status = -EBADMSG; // Indicate one or more segments failed
    } else {
        resp->overall_batch_status = 0; // Mark as success for now (basic parsing and input staging done)
    }

    // POC Cleanup for allocated input buffers
    for (uint32_t i = 0; i < decrypted_manifest.num_segments; ++i) {
        if (segment_run_info[i].enclave_host_buffer_input) {
            free(segment_run_info[i].enclave_host_buffer_input);
        }
        if (segment_run_info[i].masking_level == MASKING_AES_GCM && segment_run_info[i].enclave_device_buffer_input) {
            // Only free if it's an enclave allocated buffer for AES-GCM path.
            // DMA path uses client's pointer, not freed by service.
            cudaFree(segment_run_info[i].enclave_device_buffer_input);
        }
    }

    printf("SHARED_SERVICE_LOG: Finished basic handling of BATCH_GPU_REQUEST with overall status %d.\n", resp->overall_batch_status);
}


// --- IPC Server Functions using PAL ---


// --- IPC Server Functions using PAL ---
static PAL_HANDLE g_listening_pipe_handle = PAL_HANDLE_INITIALIZER;

int ipc_accept_client_connection(PAL_HANDLE* client_hdl, IDTYPE* client_vmid) { /* ... (unchanged) ... */ 
    if (!g_listening_pipe_handle) return -EINVAL;
    int ret = PalStreamAccept(g_listening_pipe_handle, client_hdl, NULL);
    if (ret < 0) return pal_to_unix_errno(ret);
    ret = read_exact(*client_hdl, client_vmid, sizeof(IDTYPE));
    if (ret < 0) { PalObjectDestroy(client_hdl); return ret; }
    return 0;
}
int ipc_receive_raw_message_from_client(PAL_HANDLE client_hdl, void* buf, size_t len, size_t* bytes_read) { /* ... (unchanged) ... */ 
    if (!client_hdl || !buf || !bytes_read) return -EINVAL;
    int ret = read_exact(client_hdl, buf, len);
    if (ret < 0) { *bytes_read = 0; return ret; }
    *bytes_read = len; return 0;
}
int ipc_send_raw_response_to_client(PAL_HANDLE client_hdl, const void* buf, size_t len) { /* ... (unchanged) ... */ 
    if (!client_hdl || !buf) return -EINVAL;
    if (len == 0) return 0;
    return write_exact(client_hdl, buf, len);
}

#define MAX_PAYLOAD_SIZE_FOR_SESSION ( \
    (sizeof(vector_add_request_payload_t) > sizeof(onnx_inference_request_payload_t) ? \
     ( (sizeof(vector_add_request_payload_t) > sizeof(gemm_request_payload_t)) ? \
       sizeof(vector_add_request_payload_t) : sizeof(gemm_request_payload_t) ) : \
     ( (sizeof(onnx_inference_request_payload_t) > sizeof(gemm_request_payload_t)) ? \
       sizeof(onnx_inference_request_payload_t) : sizeof(gemm_request_payload_t) ) \
    ) > sizeof(legacy_data_request_t) ? \
    ( (sizeof(vector_add_request_payload_t) > sizeof(onnx_inference_request_payload_t) ? \
      ( (sizeof(vector_add_request_payload_t) > sizeof(gemm_request_payload_t)) ? \
        sizeof(vector_add_request_payload_t) : sizeof(gemm_request_payload_t) ) : \
      ( (sizeof(onnx_inference_request_payload_t) > sizeof(gemm_request_payload_t)) ? \
        sizeof(onnx_inference_request_payload_t) : sizeof(gemm_request_payload_t) ) \
     ) ) : \
    sizeof(legacy_data_request_t) \
)

void handle_client_session(PAL_HANDLE client_handle, IDTYPE client_vmid) {
    printf("SHARED_SERVICE_LOG: New session for client VMID %u (handle %p)\n", client_vmid, client_handle);
    char ipc_buffer[sizeof(libos_ipc_msg_header_t) + MAX_PAYLOAD_SIZE_FOR_SESSION]; 
    char ipc_response_buffer[sizeof(libos_ipc_msg_header_t) + MAX_PAYLOAD_SIZE_FOR_SESSION];
    libos_ipc_msg_t* req_msg = (libos_ipc_msg_t*)ipc_buffer;
    libos_ipc_msg_t* resp_msg = (libos_ipc_msg_t*)ipc_response_buffer;
    int ret; size_t recvd_sz;

    while (1) {
        memset(ipc_buffer, 0, sizeof(ipc_buffer));
        memset(ipc_response_buffer, 0, sizeof(ipc_response_buffer));
        ret = ipc_receive_raw_message_from_client(client_handle, req_msg, sizeof(libos_ipc_msg_header_t), &recvd_sz);
        if (ret < 0 || recvd_sz != sizeof(libos_ipc_msg_header_t)) { break; }
        operation_type_t op_type = (operation_type_t)GET_UNALIGNED(req_msg->header.code);
        size_t payload_sz = GET_UNALIGNED(req_msg->header.size) - sizeof(libos_ipc_msg_header_t);
        if (payload_sz > 0) {
            if (payload_sz > MAX_PAYLOAD_SIZE_FOR_SESSION) { break; }
            ret = ipc_receive_raw_message_from_client(client_handle, req_msg->data, payload_sz, &recvd_sz);
            if (ret < 0 || recvd_sz != payload_sz) { break; }
        }
        
        if (op_type == VECTOR_ADD_REQUEST) {
            if (payload_sz != sizeof(vector_add_request_payload_t)) { break; }
            vector_add_response_payload_t va_resp; memset(&va_resp, 0, sizeof(va_resp));
            handle_vector_add_request((vector_add_request_payload_t*)req_msg->data, &va_resp);
            init_ipc_msg(resp_msg, VECTOR_ADD_REQUEST, sizeof(libos_ipc_msg_header_t) + sizeof(va_resp));
            memcpy(resp_msg->data, &va_resp, sizeof(va_resp));
            ret = ipc_send_raw_response_to_client(client_handle, resp_msg, GET_UNALIGNED(resp_msg->header.size));
        } else if (op_type == ONNX_INFERENCE_REQUEST) {
            if (payload_sz != sizeof(onnx_inference_request_payload_t)) { break; }
            onnx_inference_response_payload_t onnx_resp; memset(&onnx_resp, 0, sizeof(onnx_resp));
            handle_onnx_inference_request((onnx_inference_request_payload_t*)req_msg->data, &onnx_resp);
            init_ipc_msg(resp_msg, ONNX_INFERENCE_REQUEST, sizeof(libos_ipc_msg_header_t) + sizeof(onnx_resp));
            memcpy(resp_msg->data, &onnx_resp, sizeof(onnx_resp));
            ret = ipc_send_raw_response_to_client(client_handle, resp_msg, GET_UNALIGNED(resp_msg->header.size));
        } else if (op_type == GEMM_REQUEST) {
            if (payload_sz != sizeof(gemm_request_payload_t)) { break; }
            gemm_response_payload_t gemm_resp; memset(&gemm_resp, 0, sizeof(gemm_resp));
            handle_gemm_request((gemm_request_payload_t*)req_msg->data, &gemm_resp);
            init_ipc_msg(resp_msg, GEMM_REQUEST, sizeof(libos_ipc_msg_header_t) + sizeof(gemm_resp));
            memcpy(resp_msg->data, &gemm_resp, sizeof(gemm_resp));
            ret = ipc_send_raw_response_to_client(client_handle, resp_msg, GET_UNALIGNED(resp_msg->header.size));
        } else if (op_type == BATCH_GPU_REQUEST) { // Using non-POC enum BATCH_GPU_REQUEST
            if (payload_sz != sizeof(batch_gpu_request_payload_t)) { // Using non-POC struct
                fprintf(stderr, "SHARED_SERVICE_ERROR: BATCH_GPU_REQUEST payload size mismatch (got %zu, expected %zu).\n", payload_sz, sizeof(batch_gpu_request_payload_t));
                break; 
            }
            batch_gpu_response_payload_t batch_resp; memset(&batch_resp, 0, sizeof(batch_resp)); // non-POC struct
            handle_batch_gpu_request((batch_gpu_request_payload_t*)req_msg->data, &batch_resp); // non-POC struct
            init_ipc_msg(resp_msg, BATCH_GPU_REQUEST, sizeof(libos_ipc_msg_header_t) + sizeof(batch_resp));
            memcpy(resp_msg->data, &batch_resp, sizeof(batch_resp));
            ret = ipc_send_raw_response_to_client(client_handle, resp_msg, GET_UNALIGNED(resp_msg->header.size));
        } else if (op_type == STORE_DATA || op_type == RETRIEVE_DATA) {
            if (payload_sz > sizeof(legacy_data_request_t)) { break; }
            legacy_data_response_t legacy_resp; memset(&legacy_resp, 0, sizeof(legacy_resp));
            if (op_type == STORE_DATA) handle_legacy_store_request((legacy_data_request_t*)req_msg->data, &legacy_resp);
            else handle_legacy_retrieve_request((legacy_data_request_t*)req_msg->data, &legacy_resp);
            init_ipc_msg(resp_msg, op_type, sizeof(libos_ipc_msg_header_t) + sizeof(legacy_resp));
            memcpy(resp_msg->data, &legacy_resp, sizeof(legacy_resp));
            ret = ipc_send_raw_response_to_client(client_handle, resp_msg, GET_UNALIGNED(resp_msg->header.size));
        } else { ret = -1; }
        if (ret < 0) { break; }
    }
    PalObjectDestroy(&client_handle); 
    printf("SHARED_SERVICE_LOG: Session ended for client VMID %u.\n", client_vmid);
}

int main(int argc, char *argv[]) {
    if (argc > 1 && strcmp(argv[1], "--run-tests") == 0) {
        if (mkdir(UNTRUSTED_SHM_PATH_PREFIX, 0775) != 0 && errno != EEXIST) {
            perror("SHARED_SERVICE_TEST_FATAL: mkdir for UNTRUSTED_SHM_PATH_PREFIX failed"); return EXIT_FAILURE;
        }
        if (init_onnx_runtime(DEFAULT_ONNX_MODEL_PATH) != 0) { 
            fprintf(stderr, "SHARED_SERVICE_TEST_FATAL: ONNX init failed for tests.\n"); /* Don't exit yet, cuBLAS might be testable */
        }
        if (init_cublas() != 0) {
            fprintf(stderr, "SHARED_SERVICE_TEST_FATAL: cuBLAS init failed for tests.\n");
        }
        int test_ret = run_service_logic_tests();
        shutdown_cublas();
        shutdown_onnx_runtime(); 
        return test_ret;
    }
    printf("SHARED_SERVICE_LOG: Starting Data Storage Service...\n");
    if (init_ipc() < 0) { fprintf(stderr, "FATAL: init_ipc failed.\n"); return EXIT_FAILURE; }
    if (init_onnx_runtime(DEFAULT_ONNX_MODEL_PATH) != 0) { 
        fprintf(stderr, "FATAL: init_onnx_runtime failed.\n"); /* Continue to allow other services */
    }
    if (init_cublas() != 0) {
        fprintf(stderr, "FATAL: init_cublas failed.\n"); /* Continue to allow other services */
    }
    char listening_uri[PIPE_URI_SIZE];
    if (snprintf(listening_uri, sizeof(listening_uri), URI_PREFIX_PIPE "%lu/%u", g_pal_public_state->instance_id, g_process_ipc_ids.self_vmid) < 0) {
        fprintf(stderr, "FATAL: snprintf for listening_uri failed.\n"); shutdown_cublas(); shutdown_onnx_runtime(); return EXIT_FAILURE;
    }
    if (PalStreamListen(listening_uri, PAL_LISTEN_DEFAULT, &g_listening_pipe_handle) < 0) {
        fprintf(stderr, "FATAL: PalStreamListen failed for URI %s.\n", listening_uri); shutdown_cublas(); shutdown_onnx_runtime(); return EXIT_FAILURE;
    }
    if (mkdir(UNTRUSTED_SHM_PATH_PREFIX, 0775) != 0 && errno != EEXIST) {
        perror("FATAL: mkdir for UNTRUSTED_SHM_PATH_PREFIX failed"); 
        if (g_listening_pipe_handle) PalObjectDestroy(&g_listening_pipe_handle); 
        shutdown_cublas(); shutdown_onnx_runtime(); return EXIT_FAILURE;
    }
    printf("SHARED_SERVICE_LOG: Listening on %s (handle %p). Entering main loop...\n", listening_uri, g_listening_pipe_handle);
    while (1) {
        PAL_HANDLE client_hdl = PAL_HANDLE_INITIALIZER; IDTYPE client_vmid = 0;
        if (ipc_accept_client_connection(&client_hdl, &client_vmid) < 0) continue;
        handle_client_session(client_hdl, client_vmid);
    }
    if (g_listening_pipe_handle) PalObjectDestroy(&g_listening_pipe_handle); 
    shutdown_cublas();
    shutdown_onnx_runtime();
    return EXIT_SUCCESS;
}

// --- Service Logic Unit Tests ---
// (test_shm_store_retrieve, test_gpu_store_placeholder, test_gpu_retrieve_placeholder, run_service_logic_tests remain unchanged from previous version)
static int test_shm_store_retrieve() {
    printf("\nRunning test_shm_store_retrieve...\n");
    legacy_data_request_t store_req; 
    legacy_data_response_t store_resp; 
    legacy_data_request_t retrieve_req; 
    legacy_data_response_t retrieve_resp; 

    const char* test_filename = "unit_test_shm_file.txt";
    const char* test_content = "Hello SHM from unit test!";
    size_t test_content_len = strlen(test_content);

    memset(&store_req, 0, sizeof(store_req));
    store_req.operation = STORE_DATA;
    store_req.sensitivity = SENSITIVITY_LOW_SHM;
    strncpy(store_req.path, test_filename, MAX_LEGACY_PATH_SIZE -1); 
    store_req.data_size = test_content_len;
    memcpy(store_req.data, test_content, test_content_len);

    printf("  Calling handle_legacy_store_request for SHM...\n"); 
    handle_legacy_store_request(&store_req, &store_resp); 
    assert(store_resp.status == 0 && "SHM store request failed");
    if (store_resp.status != 0) { fprintf(stderr, "  SHM store failed with status: %d\n", store_resp.status); return -1; }

    char full_shm_path[MAX_LEGACY_PATH_SIZE + sizeof(UNTRUSTED_SHM_PATH_PREFIX) + 1]; 
    snprintf(full_shm_path, sizeof(full_shm_path), "%s/%s", UNTRUSTED_SHM_PATH_PREFIX, test_filename);
    FILE* fp = fopen(full_shm_path, "rb");
    assert(fp != NULL && "Failed to open SHM file for verification");
    if (!fp) { perror("  fopen for SHM verification failed"); return -1; }
    char read_buffer[100]; memset(read_buffer, 0, sizeof(read_buffer));
    size_t bytes_read = fread(read_buffer, 1, sizeof(read_buffer) -1, fp); fclose(fp);
    assert(bytes_read == test_content_len && "SHM file content length mismatch");
    assert(memcmp(test_content, read_buffer, test_content_len) == 0 && "SHM file content mismatch");

    memset(&retrieve_req, 0, sizeof(retrieve_req));
    retrieve_req.operation = RETRIEVE_DATA;
    retrieve_req.sensitivity = SENSITIVITY_LOW_SHM;
    strncpy(retrieve_req.path, test_filename, MAX_LEGACY_PATH_SIZE -1); 
    printf("  Calling handle_legacy_retrieve_request for SHM...\n"); 
    handle_legacy_retrieve_request(&retrieve_req, &retrieve_resp); 
    assert(retrieve_resp.status == 0 && "SHM retrieve request failed");
    if (retrieve_resp.status != 0) { fprintf(stderr, "  SHM retrieve failed with status: %d\n", retrieve_resp.status); return -1; }
    assert(retrieve_resp.data_size == test_content_len && "Retrieved SHM data length mismatch");
    assert(memcmp(test_content, retrieve_resp.data, test_content_len) == 0 && "Retrieved SHM data content mismatch");
    if (remove(full_shm_path) != 0) { perror("  Failed to remove SHM test file"); }
    printf("test_shm_store_retrieve: PASSED\n"); return 0;
}

static int test_gpu_store_placeholder() {
    printf("\nRunning test_gpu_store_placeholder...\n");
    legacy_data_request_t req; 
    legacy_data_response_t resp; 
    memset(&req, 0, sizeof(req));
    req.operation = STORE_DATA;
    req.sensitivity = SENSITIVITY_MEDIUM_GPU;
    strncpy(req.path, "gpu_test_id_1", MAX_LEGACY_PATH_SIZE -1); 
    req.data_size = 10; memset(req.data, 'G', 10);
    printf("  Calling handle_legacy_store_request for GPU (placeholder)...\n"); 
    handle_legacy_store_request(&req, &resp); 
    assert(resp.status == 0 && "GPU store placeholder request failed");
    if (resp.status != 0) { fprintf(stderr, "  GPU store placeholder failed with status: %d\n", resp.status); return -1; }
    printf("test_gpu_store_placeholder: PASSED\n"); return 0;
}

static int test_gpu_retrieve_placeholder() {
    printf("\nRunning test_gpu_retrieve_placeholder...\n");
    legacy_data_request_t req; 
    legacy_data_response_t resp; 
    memset(&req, 0, sizeof(req));
    req.operation = RETRIEVE_DATA;
    req.sensitivity = SENSITIVITY_MEDIUM_GPU;
    strncpy(req.path, "gpu_test_id_2", MAX_LEGACY_PATH_SIZE-1); 
    printf("  Calling handle_legacy_retrieve_request for GPU (placeholder)...\n"); 
    handle_legacy_retrieve_request(&req, &resp); 
    assert(resp.status == 0 && "GPU retrieve placeholder request failed");
    if (resp.status != 0) { fprintf(stderr, "  GPU retrieve placeholder failed with status: %d\n", resp.status); return -1; }
    printf("test_gpu_retrieve_placeholder: PASSED\n"); return 0;
}

static int run_service_logic_tests() {
    printf("\n--- Running Shared Service Logic Unit Tests ---\n");
    int overall_status = 0;
    if (test_shm_store_retrieve() != 0) overall_status = -1;
    if (test_gpu_store_placeholder() != 0) overall_status = -1;
    if (test_gpu_retrieve_placeholder() != 0) overall_status = -1;
    // TODO: Add a test for ONNX inference logic if possible (might need model, etc.)
    // TODO: Add a test for GEMM logic if possible.
    if (overall_status == 0) printf("\nAll shared_service logic tests PASSED.\n");
    else printf("\nOne or more shared_service logic tests FAILED.\n");
    printf("--- Finished Shared Service Logic Unit Tests ---\n");
    return overall_status;
}
