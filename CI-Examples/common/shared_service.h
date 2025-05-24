#ifndef SHARED_SERVICE_H
#define SHARED_SERVICE_H

#include <stdint.h> // For uint32_t

// Max number of float elements in a single array for vector addition example
#define VECTOR_ARRAY_MAX_ELEMENTS 1024 
// Fixed size for this specific vector addition example
#define VECTOR_ARRAY_DEFAULT_ELEMENTS 128 

#define MAX_LEGACY_PATH_SIZE 256 // For legacy STORE/RETRIEVE operations

// Constants from libos_aes_gcm.h (should match)
#define GCM_KEY_SIZE_BYTES 32
#define GCM_IV_SIZE_BYTES 12
#define GCM_TAG_SIZE_BYTES 16

typedef enum {
    STORE_DATA,         // Legacy operation for simple store
    RETRIEVE_DATA,      // Legacy operation for simple retrieve
    VECTOR_ADD_REQUEST, // New operation type for vector addition
    ONNX_INFERENCE_REQUEST, // New operation type for ONNX inference
    GEMM_REQUEST        // New operation type for SGEMM
} operation_type_t;

// This enum might still be useful for the legacy STORE_DATA/RETRIEVE_DATA operations
// if the shared_service continues to support them alongside VECTOR_ADD.
typedef enum {
    SENSITIVITY_MEDIUM_GPU, // Data processed by GPU, implies masking needed by service
    SENSITIVITY_LOW_SHM     // Data for untrusted shared memory (raw)
} data_sensitivity_t;


// --- Structures for Legacy STORE_DATA / RETRIEVE_DATA operations ---
// These are kept for potential backward compatibility or other service functions.
// For VECTOR_ADD, we will use more specific structures.

typedef struct {
    operation_type_t operation;
    data_sensitivity_t sensitivity;
    char path[MAX_LEGACY_PATH_SIZE]; // Identifier for GPU data or relative path for SHM
    uint32_t data_size; // Size in bytes of the data payload
    // Max payload size for these legacy operations
    unsigned char data[VECTOR_ARRAY_MAX_ELEMENTS * sizeof(float)]; 
} legacy_data_request_t;

typedef struct {
    int status; // 0 for success, negative errno for errors
    uint32_t data_size; // Size in bytes of the retrieved data
    unsigned char data[VECTOR_ARRAY_MAX_ELEMENTS * sizeof(float)]; 
} legacy_data_response_t;


// --- Structures specific to VECTOR_ADD_REQUEST operation ---

// Enum to specify the masking level of GPU-related data
typedef enum {
    MASKING_AES_GCM, // Data is encrypted and authenticated using AES-GCM
    MASKING_NONE     // Data is plaintext (e.g., for low sensitivity or if channel is otherwise secured)
} gpu_data_masking_level_t;

typedef struct {
    gpu_data_masking_level_t masking_level; // Indicates if data_b and data_c are masked
    uint32_t array_len_elements; // Number of float elements in each array (B and C)

    // Input Array B. Content is plaintext if masking_level is MASKING_NONE,
    // otherwise it's AES-GCM ciphertext.
    // IV and Tag for B are only meaningful if masking_level is MASKING_AES_GCM.
    unsigned char iv_b[GCM_IV_SIZE_BYTES];
    unsigned char tag_b[GCM_TAG_SIZE_BYTES];
    unsigned char data_b[VECTOR_ARRAY_MAX_ELEMENTS * sizeof(float)]; // Name changed from masked_data_b

    // Input Array C. Content is plaintext if masking_level is MASKING_NONE.
    // IV and Tag for C are only meaningful if masking_level is MASKING_AES_GCM.
    unsigned char iv_c[GCM_IV_SIZE_BYTES];
    unsigned char tag_c[GCM_TAG_SIZE_BYTES];
    unsigned char data_c[VECTOR_ARRAY_MAX_ELEMENTS * sizeof(float)]; // Name changed from masked_data_c

    // DMA related fields: if masking_level is MASKING_NONE and these are non-zero,
    // the service will use these as source device pointers for DMA from client's
    // host-allocated pinned memory.
    uint64_t src_device_ptr_b; 
    uint64_t src_device_ptr_c;

} vector_add_request_payload_t;

typedef struct {
    int status; // 0 for success, error code otherwise (e.g., CUDA error from service)
    gpu_data_masking_level_t masking_level; // Indicates if data_a is masked
    uint32_t array_len_elements; // Number of float elements in the result array A

    // Result Array A. Content is plaintext if masking_level is MASKING_NONE.
    // IV and Tag for A are only meaningful if masking_level is MASKING_AES_GCM.
    unsigned char iv_a[GCM_IV_SIZE_BYTES];
    unsigned char tag_a[GCM_TAG_SIZE_BYTES];
    unsigned char data_a[VECTOR_ARRAY_MAX_ELEMENTS * sizeof(float)]; // Name changed from masked_data_a

} vector_add_response_payload_t;


// --- Structures specific to ONNX_INFERENCE_REQUEST operation ---

// For MobileNetV2, input is typically 1x3x224x224 floats
// 1 * 3 * 224 * 224 * 4 bytes = 602,112 bytes.
// Let's define a slightly larger buffer for flexibility or other similar models.
#define MAX_ONNX_INPUT_SIZE_BYTES (610000) 

// For MobileNetV2, output is typically 1x1000 floats (1000 classes)
// 1000 * 4 bytes = 4000 bytes.
#define MAX_ONNX_OUTPUT_SIZE_BYTES (4100)
#define MAX_MODEL_NAME_LEN 128 // Max length for model name string

typedef struct {
    // Optional: Model identifier if the service supports multiple models
    // char model_name[MAX_MODEL_NAME_LEN]; 

    gpu_data_masking_level_t masking_level; // Indicates if input_tensor is masked
    uint32_t input_tensor_size_bytes; // Actual size of the input tensor data

    // IV and Tag are only meaningful if masking_level is MASKING_AES_GCM.
    unsigned char iv[GCM_IV_SIZE_BYTES];
    unsigned char tag[GCM_TAG_SIZE_BYTES];
    // Input tensor data. Content is plaintext if masking_level is MASKING_NONE.
    unsigned char input_tensor[MAX_ONNX_INPUT_SIZE_BYTES]; // Name changed from masked_input_tensor

    // DMA related field: if masking_level is MASKING_NONE and this is non-zero,
    // the service will use this as source device pointer for DMA from client's
    // host-allocated pinned memory.
    uint64_t src_device_ptr_input_tensor;

} onnx_inference_request_payload_t;

typedef struct {
    int status; // 0 for success, error code otherwise (e.g., ONNX runtime error)
    gpu_data_masking_level_t masking_level; // Indicates if output_tensor is masked
    uint32_t output_tensor_size_bytes; // Actual size of the output tensor data

    // IV and Tag are only meaningful if masking_level is MASKING_AES_GCM.
    unsigned char iv[GCM_IV_SIZE_BYTES];
    unsigned char tag[GCM_TAG_SIZE_BYTES];
    // Output tensor data. Content is plaintext if masking_level is MASKING_NONE.
    unsigned char output_tensor[MAX_ONNX_OUTPUT_SIZE_BYTES]; // Name changed from masked_output_tensor

} onnx_inference_response_payload_t;


// --- Structures specific to GEMM_REQUEST operation ---

// Example: For a 2048x2048 matrix of floats: 2048 * 2048 * 4 bytes = 16,777,216 bytes
// This might be too large for a single IPC message payload depending on Gramine's limits
// and stack space if these structs are stack-allocated.
// Consider smaller max dimensions for this example, or chunking for larger matrices.
// Let's use a more modest max, e.g., 512x512 floats = 1MB.
// Or 256*256 floats = 262144 bytes.
// For now, let's use 2048*2048 elements as a placeholder for size calculation,
// but note that the actual data arrays might be smaller in the struct to fit IPC.
// Let's define MAX_GEMM_DIM for one dimension, e.g., 512.
// So, max elements = 512*512 = 262144. Max bytes = 262144 * 4 = 1,048,576 bytes.
#define MAX_GEMM_DIM_SIZE 512
#define MAX_GEMM_MATRIX_ELEMENTS (MAX_GEMM_DIM_SIZE * MAX_GEMM_DIM_SIZE)
#define MAX_GEMM_MATRIX_SIZE_BYTES (MAX_GEMM_MATRIX_ELEMENTS * sizeof(float))

typedef struct {
    int M, N, K; // Matrix dimensions: A (M x K), B (K x N), C (M x N)

    uint32_t matrix_a_size_bytes; // M*K*sizeof(float)
    unsigned char iv_a[GCM_IV_SIZE_BYTES];
    unsigned char tag_a[GCM_TAG_SIZE_BYTES];
    // Actual data size sent will be matrix_a_size_bytes.
    // Content is AES-GCM ciphertext if masking_level is MASKING_AES_GCM,
    // otherwise plaintext (if not using DMA path).
    unsigned char matrix_a[MAX_GEMM_MATRIX_SIZE_BYTES]; 

    uint32_t matrix_b_size_bytes; // K*N*sizeof(float)
    unsigned char iv_b[GCM_IV_SIZE_BYTES];
    unsigned char tag_b[GCM_TAG_SIZE_BYTES];
    // Actual data size sent will be matrix_b_size_bytes.
    // Content is AES-GCM ciphertext if masking_level is MASKING_AES_GCM,
    // otherwise plaintext (if not using DMA path).
    unsigned char matrix_b[MAX_GEMM_MATRIX_SIZE_BYTES];

    // DMA related fields: if masking_level is MASKING_NONE and these are non-zero,
    // the service will use these as source device pointers for DMA from client's
    // host-allocated pinned memory.
    uint64_t src_device_ptr_matrix_a;
    uint64_t src_device_ptr_matrix_b;
    
} gemm_request_payload_t;

typedef struct {
    int status; // 0 for success, error code otherwise (e.g., cuBLAS error)
    uint32_t matrix_c_size_bytes; // M*N*sizeof(float)

    unsigned char iv_c[GCM_IV_SIZE_BYTES];
    unsigned char tag_c[GCM_TAG_SIZE_BYTES];
    // Actual data size received will be matrix_c_size_bytes
    unsigned char masked_matrix_c[MAX_GEMM_MATRIX_SIZE_BYTES];

} gemm_response_payload_t;


/*
 * The actual message sent over IPC will be `libos_ipc_msg` from `libos_ipc.h`.
 * The `libos_ipc_msg.data` field will point to one of the payload structures above,
 * depending on `libos_ipc_msg.header.code` which should map to `operation_type_t`.
 * The `libos_ipc_msg.header.size` must be set to:
 *   sizeof(libos_ipc_msg_header_t) + sizeof(relevant_payload_struct)
 *
 * Example for sending a vector_add request:
 *   struct libos_ipc_msg msg_to_send;
 *   vector_add_request_payload_t vec_add_payload;
 *   onnx_inference_request_payload_t onnx_payload;
 *   gemm_request_payload_t gemm_payload;
 *
 *   // Example for VECTOR_ADD_REQUEST:
 *   // ... populate vec_add_payload ...
 *   init_ipc_msg(&msg_to_send, VECTOR_ADD_REQUEST, sizeof(libos_ipc_msg_header_t) + sizeof(vec_add_payload));
 *   memcpy(msg_to_send.data, &vec_add_payload, sizeof(vec_add_payload));
 *   ipc_send_msg_and_get_response(dest_vmid, &msg_to_send, &response_msg_ptr);
 *
 *   // Example for ONNX_INFERENCE_REQUEST:
 *   // ... populate onnx_payload ...
 *   init_ipc_msg(&msg_to_send, ONNX_INFERENCE_REQUEST, sizeof(libos_ipc_msg_header_t) + sizeof(onnx_payload));
 *   memcpy(msg_to_send.data, &onnx_payload, sizeof(onnx_payload));
 *   ipc_send_msg_and_get_response(dest_vmid, &msg_to_send, &response_msg_ptr);
 *
 *   // Example for GEMM_REQUEST:
 *   // ... populate gemm_payload ...
 *   init_ipc_msg(&msg_to_send, GEMM_REQUEST, sizeof(libos_ipc_msg_header_t) + sizeof(gemm_payload));
 *   memcpy(msg_to_send.data, &gemm_payload, sizeof(gemm_payload));
 *   ipc_send_msg_and_get_response(dest_vmid, &msg_to_send, &response_msg_ptr);
 */

#endif // SHARED_SERVICE_H
