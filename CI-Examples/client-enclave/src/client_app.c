#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>    // For srand
#include <assert.h>
#include <math.h>    // For fabs
#include <cuda_runtime.h> // For CUDA host memory allocation

// Gramine includes - assumed to be in include path
#include "libos_ipc.h"
#include "libos_aes_gcm.h"

// Shared header with service
#include "shared_service.h" // From CI-Examples/common/

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA_ERROR at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return EXIT_FAILURE; \
        } \
    } while (0)

// Helper to print byte arrays for debugging
static void print_float_array(const char* label, const float* data, size_t num_elements) {
    printf("%s: [", label);
    for (size_t i = 0; i < num_elements; ++i) {
        printf("%.2f", data[i]);
        if (i < num_elements - 1) printf(", ");
    }
    printf("]\n");
}

// Helper to generate random float arrays
static void generate_float_array(float* arr, size_t num_elements) {
    for (size_t i = 0; i < num_elements; ++i) {
        arr[i] = (float)(rand() % 1000) / 100.0f; // Random floats between 0.0 and 10.0
    }
}

// Helper to generate random IVs
static void generate_iv(unsigned char* iv, size_t iv_len) {
    for (size_t i = 0; i < iv_len; ++i) {
        iv[i] = (unsigned char)(rand() % 256);
    }
}


int main(int argc, char *argv[]) {
    printf("CLIENT_APP_LOG: Starting Vector Addition client application...\n");

    gpu_data_masking_level_t current_masking_level = MASKING_AES_GCM; // Default
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "--masking") == 0 && (i + 1 < argc)) {
                if (strcmp(argv[i+1], "none") == 0) {
                    current_masking_level = MASKING_NONE;
                } else if (strcmp(argv[i+1], "aes_gcm") == 0) {
                    current_masking_level = MASKING_AES_GCM;
                } else {
                    fprintf(stderr, "CLIENT_APP_WARNING: Unknown masking mode '%s'. Defaulting to AES-GCM.\n", argv[i+1]);
                }
                i++; // Skip next argument as it has been consumed
            }
        }
    }
    printf("CLIENT_APP_LOG: Using masking mode: %s\n",
           current_masking_level == MASKING_AES_GCM ? "AES-GCM" : "None");

    unsigned char aes_key[GCM_KEY_SIZE_BYTES];
    if (current_masking_level == MASKING_AES_GCM) {
        // Initialize key only if masking is enabled
        for(int i=0; i < GCM_KEY_SIZE_BYTES; ++i) aes_key[i] = (unsigned char)(i + 0x55); // Example key
    }
    srand(time(NULL)); 

    // 1. Initialize input data
    const int num_elements = VECTOR_ARRAY_DEFAULT_ELEMENTS;
    size_t data_size_bytes = num_elements * sizeof(float);

    float *h_B = NULL;
    float *h_C = NULL;
    float *h_A_result = NULL; // Will be dynamically allocated
    // h_A_expected can remain stack-allocated as it's purely for client-side verification
    float h_A_expected[num_elements]; 

    uint64_t d_ptr_B = 0;
    uint64_t d_ptr_C = 0;

    if (current_masking_level == MASKING_NONE) {
        printf("CLIENT_APP_LOG: Allocating page-locked (pinned) host memory for B and C using cudaHostAlloc...\n");
        CUDA_CHECK(cudaHostAlloc((void**)&h_B, data_size_bytes, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc((void**)&h_C, data_size_bytes, cudaHostAllocDefault));

        printf("CLIENT_APP_LOG: Getting device pointers for h_B and h_C...\n");
        CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_ptr_B, (void*)h_B, 0));
        CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_ptr_C, (void*)h_C, 0));
        printf("CLIENT_APP_LOG: Device pointer for h_B: 0x%lx\n", d_ptr_B);
        printf("CLIENT_APP_LOG: Device pointer for h_C: 0x%lx\n", d_ptr_C);

    } else { // MASKING_AES_GCM
        // Using standard malloc for non-DMA path as per instruction
        h_B = (float*)malloc(data_size_bytes);
        h_C = (float*)malloc(data_size_bytes);
        if (!h_B || !h_C) {
            fprintf(stderr, "CLIENT_APP_ERROR: Failed to allocate memory for h_B or h_C.\n");
            if(h_B) free(h_B);
            if(h_C) free(h_C);
            return EXIT_FAILURE;
        }
    }

    generate_float_array(h_B, num_elements);
    generate_float_array(h_C, num_elements);

    for (int i = 0; i < num_elements; ++i) {
        h_A_expected[i] = h_B[i] + h_C[i];
    }

    // Allocate h_A_result based on masking level BEFORE preparing request
    if (current_masking_level == MASKING_NONE) {
        printf("CLIENT_APP_LOG: Allocating page-locked (pinned) host memory for h_A_result using cudaHostAlloc...\n");
        CUDA_CHECK(cudaHostAlloc((void**)&h_A_result, data_size_bytes, cudaHostAllocDefault));
        if (!h_A_result) {
            fprintf(stderr, "CLIENT_APP_ERROR: cudaHostAlloc failed for h_A_result.\n");
            // Go to cleanup that handles h_B and h_C
            goto cleanup_va_pre_ipc;
        }
    } else { // MASKING_AES_GCM
        h_A_result = (float*)malloc(data_size_bytes);
        if (!h_A_result) {
            fprintf(stderr, "CLIENT_APP_ERROR: Failed to allocate memory for h_A_result (AES-GCM).\n");
            // Go to cleanup that handles h_B and h_C
            goto cleanup_va_pre_ipc;
        }
    }
    // print_float_array("Host B (plaintext)", h_B, num_elements);
    // print_float_array("Host C (plaintext)", h_C, num_elements);
    // print_float_array("Host A_expected (plaintext)", h_A_expected, num_elements);


    // 2. Prepare request payload
    vector_add_request_payload_t request_payload;
    memset(&request_payload, 0, sizeof(request_payload)); // Important to zero out
    request_payload.array_len_elements = num_elements;
    request_payload.masking_level = current_masking_level;
    int ret;


    if (current_masking_level == MASKING_AES_GCM) {
        printf("CLIENT_APP_LOG: Encrypting input arrays B and C using AES-GCM...\n");
        request_payload.src_device_ptr_b = 0; 
        request_payload.src_device_ptr_c = 0; 
        request_payload.dest_host_ptr_a = 0; // No DMA for output in AES-GCM
        request_payload.dest_host_buffer_size_a = 0;
        generate_iv(request_payload.iv_b, GCM_IV_SIZE_BYTES);
        ret = libos_aes_gcm_encrypt(aes_key, request_payload.iv_b,
                                    (const unsigned char*)h_B, data_size_bytes,
                                    request_payload.data_b, request_payload.tag_b, NULL, 0);
        if (ret != 0) { fprintf(stderr, "CLIENT_APP_ERROR: Encryption of array B failed: %d\n", ret); goto cleanup_va_pre_ipc; }

        generate_iv(request_payload.iv_c, GCM_IV_SIZE_BYTES);
        ret = libos_aes_gcm_encrypt(aes_key, request_payload.iv_c,
                                    (const unsigned char*)h_C, data_size_bytes,
                                    request_payload.data_c, request_payload.tag_c, NULL, 0);
        if (ret != 0) { fprintf(stderr, "CLIENT_APP_ERROR: Encryption of array C failed: %d\n", ret); goto cleanup_va_pre_ipc; }
    } else { // MASKING_NONE
        printf("CLIENT_APP_LOG: Preparing input arrays B and C as plaintext for DMA...\n");
        // Data for B and C is already in pinned h_B, h_C. Service will use device pointers.
        // No need to memcpy to request_payload.data_b/c if service only uses device_ptrs.
        // However, current service logic might still expect data_b/c if ptrs are zero for some reason,
        // but for this DMA path, we assume device_ptrs are primary.
        // Let's clear them to avoid confusion if service logic changes.
        // memset(request_payload.data_b, 0, sizeof(request_payload.data_b));
        // memset(request_payload.data_c, 0, sizeof(request_payload.data_c));

        request_payload.src_device_ptr_b = d_ptr_B;
        request_payload.src_device_ptr_c = d_ptr_C;
        printf("CLIENT_APP_LOG: Using DMA path for input. Device pointers set in request.\n");

        // Set DMA output pointers
        request_payload.dest_host_ptr_a = (uint64_t)h_A_result;
        request_payload.dest_host_buffer_size_a = data_size_bytes;
        printf("CLIENT_APP_LOG: Using DMA path for output. Dest host pointer 0x%lx, size %u set in request.\n",
               request_payload.dest_host_ptr_a, request_payload.dest_host_buffer_size_a);
    }

    // 3. Prepare IPC message
    struct libos_ipc_msg request_msg;
    // The size here is crucial: header + actual payload size
    init_ipc_msg(&request_msg, VECTOR_ADD_REQUEST, 
                 sizeof(libos_ipc_msg_header_t) + sizeof(vector_add_request_payload_t));
    
    // Copy the specific payload into the data part of libos_ipc_msg
    // Ensure data field of request_msg is large enough or dynamically allocated.
    // For this example, assuming libos_ipc_msg.data is a flexible array member or large enough.
    // A safer way would be to allocate memory for request_msg based on payload size.
    // Let's assume for now it's: `unsigned char data[MAX_PAYLOAD_SIZE];` in libos_ipc_msg
    if (sizeof(vector_add_request_payload_t) > sizeof(request_msg.data)) {
         fprintf(stderr, "CLIENT_APP_ERROR: request_payload_t size (%zu) > libos_ipc_msg.data size (%zu)\n",
                 sizeof(vector_add_request_payload_t), sizeof(request_msg.data));
         return EXIT_FAILURE;
    }
    memcpy(request_msg.data, &request_payload, sizeof(vector_add_request_payload_t));


    // 4. IPC Communication
    // TODO: Shared enclave VMID should be configurable, not hardcoded.
    IDTYPE shared_enclave_vmid = 1; 
    struct libos_ipc_msg* response_msg_ptr = NULL;

    printf("CLIENT_APP_LOG: Connecting to shared enclave (VMID: %u)...\n", shared_enclave_vmid);
    // init_ipc() should be called by Gramine LibOS early on.
    // If this client itself needs to set up connections for the first time, init_ipc()
    // might be needed here, but typically it's done by the LibOS startup.

    ret = ipc_send_msg_and_get_response(shared_enclave_vmid, &request_msg, (void**)&response_msg_ptr);
    if (ret < 0) {
        fprintf(stderr, "CLIENT_APP_ERROR: ipc_send_msg_and_get_response failed: %s\n", unix_strerror(ret));
        return EXIT_FAILURE;
    }
    if (!response_msg_ptr) {
        fprintf(stderr, "CLIENT_APP_ERROR: Received NULL response_msg_ptr from IPC.\n");
        return EXIT_FAILURE;
    }
    printf("CLIENT_APP_LOG: Received response from shared enclave.\n");

    // 5. Handle Response
    // The response_msg_ptr->data should contain the vector_add_response_payload_t
    // Check response message code and size
    if (GET_UNALIGNED(response_msg_ptr->header.code) != VECTOR_ADD_REQUEST) { // Or a new RESPONSE_VECTOR_ADD type
        fprintf(stderr, "CLIENT_APP_ERROR: Received unexpected message code: %u\n", GET_UNALIGNED(response_msg_ptr->header.code));
        ret = EXIT_FAILURE; goto cleanup_va_post_ipc;
    }
    // Expected size: header + specific payload
    size_t expected_resp_payload_size = sizeof(vector_add_response_payload_t);
    if (GET_UNALIGNED(response_msg_ptr->header.size) != (sizeof(libos_ipc_msg_header_t) + expected_resp_payload_size)) {
         fprintf(stderr, "CLIENT_APP_ERROR: Received message with unexpected size: %lu (expected payload %zu)\n",
                 GET_UNALIGNED(response_msg_ptr->header.size) - sizeof(libos_ipc_msg_header_t), expected_resp_payload_size);
        ret = EXIT_FAILURE; goto cleanup_va_post_ipc;
    }

    vector_add_response_payload_t* response_payload = (vector_add_response_payload_t*)response_msg_ptr->data;

    if (response_payload->status != 0) {
        fprintf(stderr, "CLIENT_APP_ERROR: Shared enclave reported error: %d\n", response_payload->status);
        ret = EXIT_FAILURE; goto cleanup_va_post_ipc;
    }
    if (response_payload->array_len_elements != num_elements) {
        fprintf(stderr, "CLIENT_APP_ERROR: Response array length mismatch. Expected %d, Got %u\n",
                num_elements, response_payload->array_len_elements);
        ret = EXIT_FAILURE; goto cleanup_va_post_ipc;
    }

    // Check response masking level
    if (response_payload->masking_level != current_masking_level) {
        fprintf(stderr, "CLIENT_APP_WARNING: Masking level mismatch between request (%d) and response (%d).\n",
                current_masking_level, response_payload->masking_level);
    }

    if (response_payload->masking_level == MASKING_AES_GCM) {
        printf("CLIENT_APP_LOG: Decrypting result array A (AES-GCM) into malloc'd h_A_result...\n");
        ret = libos_aes_gcm_decrypt(aes_key, response_payload->iv_a,
                                    response_payload->data_a, data_size_bytes,
                                    response_payload->tag_a, (unsigned char*)h_A_result,
                                    NULL, 0); 
        if (ret != 0) {
            fprintf(stderr, "CLIENT_APP_ERROR: Decryption of result array A failed: %d\n", ret);
            ret = EXIT_FAILURE; goto cleanup_va_post_ipc;
        }
    } else { // MASKING_NONE
        // If DMA for output was used (dest_host_ptr_a was set), data is already in h_A_result.
        // If DMA for output was NOT used by service (e.g. service older version or error),
        // then we would need to copy from response_payload->data_a.
        // For this version, we assume if client sets dest_host_ptr_a and masking is NONE,
        // the service uses DMA or fails the call. So, no memcpy needed here.
        if (request_payload.dest_host_ptr_a != 0) {
            printf("CLIENT_APP_LOG: Result array A received via DMA into pinned h_A_result.\n");
            // CUDA stream synchronization would happen here if the service used a specific stream
            // for the DtoH copy and client needed to wait on it. For now, assuming service syncs.
        } else {
            // This case should ideally not happen if service honors DMA output request.
            // If it does, it means service ignored DMA output and sent data in payload.
            printf("CLIENT_APP_LOG: Result array A received as plaintext in payload. Copying...\n");
            memcpy(h_A_result, response_payload->data_a, data_size_bytes);
        }
    }
    // print_float_array("Host A_result (processed)", h_A_result, num_elements);

    // 6. Verify Result
    printf("CLIENT_APP_LOG: Verifying results...\n");
    ret = EXIT_SUCCESS; // Assume success unless mismatches found
    int mismatches = 0;
    for (int i = 0; i < num_elements; ++i) {
        if (fabs(h_A_result[i] - h_A_expected[i]) > 1e-5) { // Tolerance for float comparison
            mismatches++;
            if (mismatches < 5) { // Print first few mismatches
                fprintf(stderr, "CLIENT_APP_ERROR: Mismatch at index %d: Expected %.5f, Got %.5f\n",
                        i, h_A_expected[i], h_A_result[i]);
            }
        }
    }

    if (mismatches == 0) {
        printf("CLIENT_APP_SUCCESS: Vector addition results verified successfully!\n");
    } else {
        fprintf(stderr, "CLIENT_APP_FAILURE: %d mismatches found in vector addition results.\n", mismatches);
        ret = EXIT_FAILURE;
        // Fall through to cleanup
    }

cleanup_va_post_ipc:
    if (response_msg_ptr) free(response_msg_ptr);

cleanup_va_pre_ipc: // Label for cleanup if error occurs before IPC call but after allocations
    if (h_A_result) {
        if (current_masking_level == MASKING_NONE) {
            printf("CLIENT_APP_LOG: Freeing page-locked (pinned) host memory for h_A_result using cudaFreeHost...\n");
            cudaError_t free_err = cudaFreeHost(h_A_result); // Changed from CUDA_CHECK for cleanup path
            if (free_err != cudaSuccess) {
                 fprintf(stderr, "CUDA_ERROR (cleanup): cudaFreeHost(h_A_result) failed: %s\n", cudaGetErrorString(free_err));
                 if (ret == EXIT_SUCCESS) ret = EXIT_FAILURE; // Ensure error is propagated
            }
        } else {
            free(h_A_result);
        }
        h_A_result = NULL;
    }

    if (current_masking_level == MASKING_NONE) {
        printf("CLIENT_APP_LOG: Freeing page-locked (pinned) host memory for B and C using cudaFreeHost...\n");
        if (h_B) { 
            cudaError_t free_err = cudaFreeHost(h_B); 
            if (free_err != cudaSuccess) {
                 fprintf(stderr, "CUDA_ERROR (cleanup): cudaFreeHost(h_B) failed: %s\n", cudaGetErrorString(free_err));
                 if (ret == EXIT_SUCCESS) ret = EXIT_FAILURE;
            }
        }
        if (h_C) {
            cudaError_t free_err = cudaFreeHost(h_C);
            if (free_err != cudaSuccess) {
                 fprintf(stderr, "CUDA_ERROR (cleanup): cudaFreeHost(h_C) failed: %s\n", cudaGetErrorString(free_err));
                 if (ret == EXIT_SUCCESS) ret = EXIT_FAILURE;
            }
        }
    } else { // MASKING_AES_GCM
        if (h_B) free(h_B);
        if (h_C) free(h_C);
    }
    h_B = NULL; 
    h_C = NULL;

    if (ret == EXIT_SUCCESS) {
        printf("CLIENT_APP_LOG: Client application finished successfully.\n");
    } else {
        printf("CLIENT_APP_LOG: Client application finished with errors.\n");
    }
    return ret;
}
