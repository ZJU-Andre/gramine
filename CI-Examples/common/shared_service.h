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
    STORE_DATA,        // Legacy operation for simple store
    RETRIEVE_DATA,     // Legacy operation for simple retrieve
    VECTOR_ADD_REQUEST // New operation type for vector addition
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

typedef struct {
    // Header fields for the specific operation, if needed, can go here.
    // For vector_add, the primary information is the arrays themselves.
    uint32_t array_len_elements; // Number of float elements in each array (B and C)

    // Input Array B (masked)
    unsigned char iv_b[GCM_IV_SIZE_BYTES];
    unsigned char tag_b[GCM_TAG_SIZE_BYTES];
    // Payload for B. Size is array_len_elements * sizeof(float)
    unsigned char masked_data_b[VECTOR_ARRAY_MAX_ELEMENTS * sizeof(float)]; 

    // Input Array C (masked)
    unsigned char iv_c[GCM_IV_SIZE_BYTES];
    unsigned char tag_c[GCM_TAG_SIZE_BYTES];
    // Payload for C. Size is array_len_elements * sizeof(float)
    unsigned char masked_data_c[VECTOR_ARRAY_MAX_ELEMENTS * sizeof(float)];

} vector_add_request_payload_t;

typedef struct {
    int status; // 0 for success, error code otherwise (e.g., CUDA error from service)
    uint32_t array_len_elements; // Number of float elements in the result array A

    // Result Array A (masked)
    unsigned char iv_a[GCM_IV_SIZE_BYTES];
    unsigned char tag_a[GCM_TAG_SIZE_BYTES];
    // Payload for A. Size is array_len_elements * sizeof(float)
    unsigned char masked_data_a[VECTOR_ARRAY_MAX_ELEMENTS * sizeof(float)];

} vector_add_response_payload_t;


/*
 * The actual message sent over IPC will be `libos_ipc_msg` from `libos_ipc.h`.
 * The `libos_ipc_msg.data` field will point to one of the payload structures above,
 * depending on `libos_ipc_msg.header.code` which should map to `operation_type_t`.
 * The `libos_ipc_msg.header.size` must be set to:
 *   sizeof(libos_ipc_msg_header_t) + sizeof(relevant_payload_struct)
 *
 * Example for sending a vector_add request:
 *   struct libos_ipc_msg msg_to_send;
 *   vector_add_request_payload_t payload;
 *   // ... populate payload ...
 *   init_ipc_msg(&msg_to_send, VECTOR_ADD_REQUEST, sizeof(libos_ipc_msg_header_t) + sizeof(payload));
 *   memcpy(msg_to_send.data, &payload, sizeof(payload));
 *   ipc_send_msg_and_get_response(dest_vmid, &msg_to_send, &response_msg_ptr);
 */

#endif // SHARED_SERVICE_H
