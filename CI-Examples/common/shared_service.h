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
    GEMM_REQUEST,        // New operation type for SGEMM
    BATCH_GPU_REQUEST,   // New operation type for batch mixed-sensitivity GPU workloads
    POC_BATCH_GPU_REQUEST // POC for new batch mixed-sensitivity GPU workloads
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

    // DMA output: if masking_level is MASKING_NONE and dest_host_ptr_a is non-zero,
    // the service will attempt to DMA the result directly to this client-provided
    // pinned host memory address. Client ensures buffer is large enough.
    uint64_t dest_host_ptr_a;
    uint32_t dest_host_buffer_size_a; // Size of the client-provided buffer for A

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

    // DMA output: if masking_level is MASKING_NONE and dest_host_ptr_output is non-zero,
    // the service will attempt to DMA the ONNX output tensor directly to this
    // client-provided pinned host memory address. Client ensures buffer is large enough.
    uint64_t dest_host_ptr_output;
    uint32_t dest_host_buffer_size_output; // Size of the client-provided buffer for output

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

    // DMA output: if masking_level is MASKING_NONE and dest_host_ptr_c is non-zero,
    // the service will attempt to DMA the result matrix C directly to this
    // client-provided pinned host memory address. Client ensures buffer is large enough.
    uint64_t dest_host_ptr_c;
    uint32_t dest_host_buffer_size_c; // Size of the client-provided buffer for C
    
} gemm_request_payload_t;

typedef struct {
    int status; // 0 for success, error code otherwise (e.g., cuBLAS error)
    uint32_t matrix_c_size_bytes; // M*N*sizeof(float)

    unsigned char iv_c[GCM_IV_SIZE_BYTES];
    unsigned char tag_c[GCM_TAG_SIZE_BYTES];
    // Actual data size received will be matrix_c_size_bytes
    unsigned char masked_matrix_c[MAX_GEMM_MATRIX_SIZE_BYTES];

} gemm_response_payload_t;


// --- Structures for POC Batch Mixed-Sensitivity GPU Workloads ---

#define MAX_SEGMENT_ID_LEN_POC 64
#define MAX_GPU_OP_ID_LEN_POC 64
#define MAX_ENCRYPTED_MANIFEST_SIZE_POC 1024
#define MAX_INLINE_DATA_SIZE_POC 256 
#define MAX_SEGMENTS_PER_BATCH_POC 4

// Simplified manifest segment for POC
typedef struct {
    char segment_id[MAX_SEGMENT_ID_LEN_POC];
    gpu_data_masking_level_t sensitivity_level_derived_masking; // Determines actual masking
    int direction; // 0:input, 1:output, 2:input/output
    char gpu_operation_id[MAX_GPU_OP_ID_LEN_POC];
    uint32_t data_size_or_max_output_size; // Expected input size or max expected output size
} poc_workload_manifest_segment_t;

// POC Encrypted Manifest (part of the batch request)
typedef struct {
    unsigned char iv[GCM_IV_SIZE_BYTES];
    unsigned char tag[GCM_TAG_SIZE_BYTES];
    uint32_t encrypted_data_size;
    unsigned char encrypted_data[MAX_ENCRYPTED_MANIFEST_SIZE_POC];
} poc_encrypted_manifest_t;

// POC IPC Segment Descriptor (part of the batch request)
typedef struct {
    char segment_id[MAX_SEGMENT_ID_LEN_POC]; // To link with manifest segment
    gpu_data_masking_level_t masking_level;  // Actual masking to apply/expect
    int direction;                           // Copied from manifest for service convenience
    uint32_t data_size;                      // Size of input data, or max expected output size for validation

    // Input specific fields
    uint64_t src_device_ptr;                 // For MASKING_NONE DMA input
    unsigned char iv_input[GCM_IV_SIZE_BYTES];     // For MASKING_AES_GCM input
    unsigned char tag_input[GCM_TAG_SIZE_BYTES];   // For MASKING_AES_GCM input
    unsigned char inline_input_data[MAX_INLINE_DATA_SIZE_POC]; // For small MASKING_AES_GCM input

    // Output specific fields (for client to specify DMA output buffer)
    uint64_t dest_host_ptr;                  // For MASKING_NONE DMA output
    uint32_t dest_buffer_size;               // Size of client's DMA output buffer
} poc_ipc_segment_descriptor_t;

// POC Batch GPU Request Payload
typedef struct {
    poc_encrypted_manifest_t encrypted_manifest;
    uint32_t num_segments; // Number of segments in the 'segments' array below
    poc_ipc_segment_descriptor_t segments[MAX_SEGMENTS_PER_BATCH_POC];
} poc_batch_gpu_request_payload_t;

// POC IPC Segment Response (part of the batch response)
typedef struct {
    char segment_id[MAX_SEGMENT_ID_LEN_POC];
    int status; // 0 for success, error code otherwise for this segment
    gpu_data_masking_level_t masking_level_of_output; // Actual masking of output_data
    uint32_t actual_output_data_size;          // Actual size of data in inline_output_data or written via DMA

    // For MASKING_AES_GCM output:
    unsigned char iv_output[GCM_IV_SIZE_BYTES];
    unsigned char tag_output[GCM_TAG_SIZE_BYTES];
    unsigned char inline_output_data[MAX_INLINE_DATA_SIZE_POC]; // For small MASKING_AES_GCM output
    // Note: For DMA output, data is written directly to client's dest_host_ptr,
    // so inline_output_data would not be used.
} poc_ipc_segment_response_t;

// POC Batch GPU Response Payload
typedef struct {
    int overall_batch_status; // 0 for overall success, or an error code
    uint32_t num_segments;    // Number of segments in the 'segment_responses' array
    poc_ipc_segment_response_t segment_responses[MAX_SEGMENTS_PER_BATCH_POC];
} poc_batch_gpu_response_payload_t;


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


// --- Structures for Batch Mixed-Sensitivity GPU Workloads ---

#define MAX_SEGMENT_ID_LEN 64
#define MAX_DATA_LOCATION_HINT_LEN 256 // Could be a path or descriptor
#define MAX_GPU_OPERATION_ID_LEN 64
#define MAX_SEGMENTS_PER_MANIFEST 16 // For POC

// Enum for data sensitivity level (distinct from masking_level which is about crypto)
typedef enum {
    SENSITIVITY_LEVEL_LOW,    // Suggests MASKING_NONE with DMA
    SENSITIVITY_LEVEL_HIGH    // Suggests MASKING_AES_GCM
    // SENSITIVITY_LEVEL_MEDIUM could be added if a distinct path is defined later
} segment_sensitivity_level_t;

// Enum for segment data direction
typedef enum {
    DATA_DIRECTION_INPUT,
    DATA_DIRECTION_OUTPUT,
    DATA_DIRECTION_INPUT_OUTPUT // For data that is modified in-place
} segment_data_direction_t;

// Structure defining a single data segment within a workload manifest
typedef struct {
    char segment_id[MAX_SEGMENT_ID_LEN];
    segment_sensitivity_level_t sensitivity_level;
    segment_data_direction_t direction;

    // Hint for client library to find/prepare data.
    // For INPUT: path to file, or identifier for pre-loaded client memory.
    // For OUTPUT: identifier for client to store/name the output.
    char data_location_hint[MAX_DATA_LOCATION_HINT_LEN];

    // GPU operation this segment is associated with.
    // Allows shared enclave to map segments to specific GPU kernels/streams.
    char gpu_operation_id[MAX_GPU_OPERATION_ID_LEN];

    // For OUTPUT segments using DMA (sensitivity_level LOW):
    // Client specifies the size of the pinned host buffer it has prepared.
    // Service will validate if actual output fits.
    uint32_t client_dma_output_buffer_size; 
    // Note: The actual client host pointer for output DMA will be in ipc_segment_descriptor_t

    // Optional: For typed data, can include hints for dimensions, datatype etc.
    // For POC, keeping it simpler. Example:
    // uint32_t expected_data_size_bytes; 
    // int dimensions[4];
    // data_type_t type_enum;

} workload_manifest_segment_t;

// Structure for the overall workload manifest
#define MAX_MANIFEST_ID_LEN 64
typedef struct {
    char manifest_id[MAX_MANIFEST_ID_LEN]; // Unique ID for this manifest version/type
    uint32_t num_segments;
    workload_manifest_segment_t segments[MAX_SEGMENTS_PER_MANIFEST];
    // uint8_t reserved[...]; // For future expansion, alignment
} workload_manifest_t;


// --- Structures for Batch IPC Communication (based on Workload Manifest) ---

// Max size for inline data in segment descriptors (e.g., for small AES-GCM payloads)
#define MAX_INLINE_SEGMENT_DATA_SIZE 256 // POC value, adjust as needed
#define MAX_ENCRYPTED_MANIFEST_SIZE 2048 // POC value for encrypted manifest

// Describes a single segment within a batch IPC request/response
typedef struct {
    char segment_id[MAX_SEGMENT_ID_LEN]; // Corresponds to workload_manifest_segment_t.segment_id
    gpu_data_masking_level_t masking_level; // MASKING_AES_GCM or MASKING_NONE for this segment

    // --- Input Data Fields ---
    // Used if direction is INPUT or INPUT_OUTPUT in the manifest

    // For MASKING_AES_GCM (input):
    unsigned char iv[GCM_IV_SIZE_BYTES];
    unsigned char tag[GCM_TAG_SIZE_BYTES];
    uint32_t encrypted_data_size; // Actual size of encrypted_data following
    // For small data, can be inlined. For larger, consider separate transfer or use DMA if appropriate.
    // For POC, allow small inline data.
    unsigned char encrypted_data[MAX_INLINE_SEGMENT_DATA_SIZE]; 

    // For MASKING_NONE with DMA (input):
    uint64_t src_device_ptr; // Client's device pointer for input data
    uint32_t input_data_size;  // Actual size of input data at src_device_ptr

    // --- Output Data Fields (passed in request for DMA output setup) ---
    // Used if direction is OUTPUT or INPUT_OUTPUT in the manifest and MASKING_NONE with DMA

    // For MASKING_NONE with DMA (output):
    uint64_t dest_host_ptr;           // Client's host pinned memory pointer for output
    uint32_t dest_buffer_size_bytes;  // Size of client's output buffer

    // GPU operation ID this segment is for (copied from manifest for service routing)
    char gpu_operation_id[MAX_GPU_OPERATION_ID_LEN];

} ipc_segment_descriptor_t;


// Batch GPU Request Payload
// This structure is sent from the client to the shared service.
typedef struct {
    // Encrypted Workload Manifest
    unsigned char encrypted_manifest_iv[GCM_IV_SIZE_BYTES];
    unsigned char encrypted_manifest_tag[GCM_TAG_SIZE_BYTES];
    uint32_t encrypted_manifest_size;
    // For POC, inline encrypted manifest. Could be a handle/reference if manifest is pre-registered.
    unsigned char encrypted_manifest_data[MAX_ENCRYPTED_MANIFEST_SIZE]; 

    uint32_t num_segments; // Number of segments described in segment_descriptors array
                           // This count includes all segments (input, output, input_output)
                           // for which descriptors are being sent.
    ipc_segment_descriptor_t segment_descriptors[MAX_SEGMENTS_PER_MANIFEST]; 
    // Contains descriptors for all segments involved.
    // For input segments, it carries the data (or pointers to it).
    // For output segments using DMA, it carries the client's destination host pointer and buffer size.
    // For output segments using AES-GCM, the corresponding descriptor here might be minimal,
    // as the actual encrypted output data will come in the ipc_segment_response_t.

} batch_gpu_request_payload_t;


// Describes a single segment's result in a batch IPC response
// This structure is part of the batch_gpu_response_payload_t sent from service to client.
typedef struct {
    char segment_id[MAX_SEGMENT_ID_LEN];
    int status; // 0 for success, error code otherwise for this segment's processing

    // For MASKING_AES_GCM (output):
    // These fields are populated if the segment (as defined in manifest) was an output
    // and used AES-GCM.
    unsigned char iv[GCM_IV_SIZE_BYTES];
    unsigned char tag[GCM_TAG_SIZE_BYTES];
    uint32_t encrypted_data_size; // Actual size of encrypted_data
    unsigned char encrypted_data[MAX_INLINE_SEGMENT_DATA_SIZE]; // For small outputs

    // For MASKING_NONE with DMA (output):
    // This field is populated if the segment was an output and used DMA.
    uint32_t actual_output_data_size; // Actual bytes written to client's dest_host_ptr

    // Other per-segment result info can be added here if needed.

} ipc_segment_response_t;


// Batch GPU Response Payload
typedef struct {
    int overall_batch_status; // 0 for overall success, or an error code for the batch processing
    uint32_t num_segments;    // Number of segments for which results/status are being returned
                              // This typically matches the number of segments that were marked as
                              // OUTPUT or INPUT_OUTPUT in the manifest and processed.
    ipc_segment_response_t segments[MAX_SEGMENTS_PER_MANIFEST]; // Array of segment responses

} batch_gpu_response_payload_t;


#endif // SHARED_SERVICE_H
