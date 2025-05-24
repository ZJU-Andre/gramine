#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>    // For srand, clock
#include <assert.h>
#include <math.h>    // For fabs
#include <cuda_runtime.h> // For CUDA host memory allocation

// Gramine includes
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
            /* For client_app_onnx, we might want to free allocated resources before exiting */ \
            /* This macro is simplified; more complex cleanup might be needed depending on where it's called */ \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ONNX MobileNetV2 specific dimensions
#define ONNX_MODEL_INPUT_CHANNELS 3
#define ONNX_MODEL_INPUT_HEIGHT 224
#define ONNX_MODEL_INPUT_WIDTH 224
#define ONNX_MODEL_OUTPUT_CLASSES 1000 // For ImageNet

// Helper to generate random float arrays
static void generate_random_float_tensor(float* arr, size_t num_elements) {
    for (size_t i = 0; i < num_elements; ++i) {
        // Generate floats between -1 and 1 for typical normalized image data
        arr[i] = ((float)rand() / (float)(RAND_MAX / 2.0f)) - 1.0f;
    }
}

// Helper to generate random IVs
static void generate_iv(unsigned char* iv, size_t iv_len) {
    for (size_t i = 0; i < iv_len; ++i) {
        iv[i] = (unsigned char)(rand() % 256);
    }
}

// Helper to print top K classes from output tensor
static void print_top_k_classes(const float* output_tensor, uint32_t num_classes, int k) {
    if (k > num_classes) k = num_classes;
    
    // Simple top-k: find max k times (not efficient, but ok for small k)
    printf("CLIENT_APP_ONNX: Top %d classes (Index: Score):\n", k);
    float* temp_scores = (float*)malloc(num_classes * sizeof(float));
    if (!temp_scores) {
        perror("Failed to allocate memory for temp_scores");
        return;
    }
    memcpy(temp_scores, output_tensor, num_classes * sizeof(float));

    for (int i = 0; i < k; ++i) {
        int max_idx = -1;
        float max_score = -__FLT_MAX__;
        for (uint32_t j = 0; j < num_classes; ++j) {
            if (temp_scores[j] > max_score) {
                max_score = temp_scores[j];
                max_idx = j;
            }
        }
        if (max_idx != -1) {
            printf("  %d: %.4f\n", max_idx, max_score);
            temp_scores[max_idx] = -__FLT_MAX__; // Mark as visited
        }
    }
    free(temp_scores);
}


int main(int argc, char *argv[]) {
    printf("CLIENT_APP_ONNX_LOG: Starting ONNX client application...\n");
    clock_t start_time, end_time;
    double cpu_time_used;

    gpu_data_masking_level_t current_masking_level = MASKING_AES_GCM; // Default
    const char* model_path_arg = DEFAULT_MODEL_PATH; // Default model path for native run, not used by client for IPC

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--masking") == 0 && (i + 1 < argc)) {
            if (strcmp(argv[i+1], "none") == 0) {
                current_masking_level = MASKING_NONE;
            } else if (strcmp(argv[i+1], "aes_gcm") == 0) {
                current_masking_level = MASKING_AES_GCM;
            } else {
                fprintf(stderr, "CLIENT_APP_ONNX_WARNING: Unknown masking mode '%s'. Defaulting to AES-GCM.\n", argv[i+1]);
            }
            i++; 
        } else {
            // Simple argument parsing: if not --masking, assume it might be a model path (though not used by this client directly)
            // This is more relevant for the native benchmark version.
            // model_path_arg = argv[i]; 
        }
    }
    printf("CLIENT_APP_ONNX_LOG: Using masking mode: %s\n",
           current_masking_level == MASKING_AES_GCM ? "AES-GCM" : "None");
    
    unsigned char aes_key[GCM_KEY_SIZE_BYTES];
    if (current_masking_level == MASKING_AES_GCM) {
        for(int i=0; i < GCM_KEY_SIZE_BYTES; ++i) aes_key[i] = (unsigned char)(i + 0xAA); // Example key
    }
    srand(time(NULL)); 

    // 1. Initialize input tensor data (dummy data for MobileNetV2 1x3x224x224)
    const uint32_t input_elements = ONNX_MODEL_INPUT_CHANNELS * ONNX_MODEL_INPUT_HEIGHT * ONNX_MODEL_INPUT_WIDTH;
    const uint32_t input_tensor_size_bytes = input_elements * sizeof(float);
    assert(input_tensor_size_bytes <= MAX_ONNX_INPUT_SIZE_BYTES);

    float* input_tensor_plaintext = NULL;
    uint64_t d_ptr_input_tensor = 0;

    if (current_masking_level == MASKING_NONE) {
        printf("CLIENT_APP_ONNX_LOG: Allocating page-locked (pinned) host memory for input_tensor_plaintext using cudaHostAlloc...\n");
        CUDA_CHECK(cudaHostAlloc((void**)&input_tensor_plaintext, input_tensor_size_bytes, cudaHostAllocDefault));
        
        printf("CLIENT_APP_ONNX_LOG: Getting device pointer for input_tensor_plaintext...\n");
        CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_ptr_input_tensor, (void*)input_tensor_plaintext, 0));
        printf("CLIENT_APP_ONNX_LOG: Device pointer for input_tensor_plaintext: 0x%lx\n", d_ptr_input_tensor);
    } else { // MASKING_AES_GCM
        input_tensor_plaintext = (float*)malloc(input_tensor_size_bytes);
        if (!input_tensor_plaintext) {
            fprintf(stderr, "CLIENT_APP_ONNX_ERROR: Failed to allocate memory for input tensor.\n");
            return EXIT_FAILURE;
        }
    }

    generate_random_float_tensor(input_tensor_plaintext, input_elements);
    printf("CLIENT_APP_ONNX_LOG: Generated dummy input tensor of size %u bytes (%u elements).\n",
           input_tensor_size_bytes, input_elements);

    // 2. Prepare request payload
    onnx_inference_request_payload_t request_payload;
    memset(&request_payload, 0, sizeof(request_payload)); // Important to zero out, especially for DMA pointers
    request_payload.input_tensor_size_bytes = input_tensor_size_bytes;
    // Optional: Populate model_name if service uses it (not used in this example's service)
    // strncpy(request_payload.model_name, "mobilenetv2-7.onnx", MAX_MODEL_NAME_LEN -1);

    request_payload.masking_level = current_masking_level;
    int ret;

    if (current_masking_level == MASKING_AES_GCM) {
        printf("CLIENT_APP_ONNX_LOG: Encrypting input tensor using AES-GCM...\n");
        request_payload.src_device_ptr_input_tensor = 0; // Not using DMA
        generate_iv(request_payload.iv, GCM_IV_SIZE_BYTES);
        ret = libos_aes_gcm_encrypt(aes_key, request_payload.iv,
                                        (const unsigned char*)input_tensor_plaintext,
                                        input_tensor_size_bytes,
                                        request_payload.input_tensor, // Changed from masked_input_tensor
                                        request_payload.tag,
                                        NULL, 0); 
        if (ret != 0) {
            fprintf(stderr, "CLIENT_APP_ONNX_ERROR: Encryption of input tensor failed: %d\n", ret);
            // free(input_tensor_plaintext) will be handled in cleanup section
            return EXIT_FAILURE; // Consider more robust cleanup if other resources were allocated
        }
    } else { // MASKING_NONE
        printf("CLIENT_APP_ONNX_LOG: Preparing input tensor for DMA path...\n");
        // For DMA, data is not copied into request_payload.input_tensor.
        // The service will use src_device_ptr_input_tensor to access it directly.
        request_payload.src_device_ptr_input_tensor = d_ptr_input_tensor;
        printf("CLIENT_APP_ONNX_LOG: Using DMA path. Device pointer set in request.\n");
        // request_payload.input_tensor field is not used by service if device_ptr is set.
    }
    // NOTE: The free(input_tensor_plaintext) that was here is moved to the main cleanup section
    // because for DMA, the buffer must remain valid until the service is done.

    // 3. Prepare IPC message
    struct libos_ipc_msg request_msg;
    init_ipc_msg(&request_msg, ONNX_INFERENCE_REQUEST, 
                 sizeof(libos_ipc_msg_header_t) + sizeof(onnx_inference_request_payload_t));
    
    if (sizeof(onnx_inference_request_payload_t) > sizeof(request_msg.data)) {
         fprintf(stderr, "CLIENT_APP_ONNX_ERROR: onnx_inference_request_payload_t size (%zu) > libos_ipc_msg.data size (%zu)\n",
                 sizeof(onnx_inference_request_payload_t), sizeof(request_msg.data));
         return EXIT_FAILURE;
    }
    memcpy(request_msg.data, &request_payload, sizeof(onnx_inference_request_payload_t));

    // 4. IPC Communication
    IDTYPE shared_enclave_vmid = 1; // TODO: Make configurable
    struct libos_ipc_msg* response_msg_ptr = NULL;

    printf("CLIENT_APP_ONNX_LOG: Sending ONNX inference request to shared enclave (VMID: %u)...\n", shared_enclave_vmid);
    
    start_time = clock(); // Start timing before IPC call

    ret = ipc_send_msg_and_get_response(shared_enclave_vmid, &request_msg, (void**)&response_msg_ptr);
    
    end_time = clock(); // End timing after IPC call returns
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;

    if (ret < 0) {
        fprintf(stderr, "CLIENT_APP_ONNX_ERROR: ipc_send_msg_and_get_response failed: %s\n", unix_strerror(ret));
        return EXIT_FAILURE;
    }
    if (!response_msg_ptr) {
        fprintf(stderr, "CLIENT_APP_ONNX_ERROR: Received NULL response_msg_ptr from IPC.\n");
        return EXIT_FAILURE;
    }
    printf("CLIENT_APP_ONNX_LOG: Received response from shared enclave. End-to-end time: %.4f seconds\n", cpu_time_used);

    // 5. Handle Response
    if (GET_UNALIGNED(response_msg_ptr->header.code) != ONNX_INFERENCE_REQUEST) { // Or a specific ONNX_RESPONSE code
        fprintf(stderr, "CLIENT_APP_ONNX_ERROR: Received unexpected message code: %u\n", GET_UNALIGNED(response_msg_ptr->header.code));
        free(response_msg_ptr);
        return EXIT_FAILURE;
    }
    size_t expected_resp_payload_size = sizeof(onnx_inference_response_payload_t);
    if (GET_UNALIGNED(response_msg_ptr->header.size) != (sizeof(libos_ipc_msg_header_t) + expected_resp_payload_size)) {
         fprintf(stderr, "CLIENT_APP_ONNX_ERROR: Received message with unexpected size: %lu (expected payload %zu)\n",
                 GET_UNALIGNED(response_msg_ptr->header.size) - sizeof(libos_ipc_msg_header_t), expected_resp_payload_size);
        free(response_msg_ptr);
        return EXIT_FAILURE;
    }

    onnx_inference_response_payload_t* response_payload = (onnx_inference_response_payload_t*)response_msg_ptr->data;

    if (response_payload->status != 0) {
        fprintf(stderr, "CLIENT_APP_ONNX_ERROR: Shared enclave reported error for ONNX inference: %d\n", response_payload->status);
        free(response_msg_ptr);
        return EXIT_FAILURE;
    }
    if (response_payload->output_tensor_size_bytes == 0 || response_payload->output_tensor_size_bytes > MAX_ONNX_OUTPUT_SIZE_BYTES) {
        fprintf(stderr, "CLIENT_APP_ONNX_ERROR: Invalid output tensor size from service: %u\n", response_payload->output_tensor_size_bytes);
        free(response_msg_ptr);
        return EXIT_FAILURE;
    }
    
    float* output_tensor_plaintext = (float*)malloc(response_payload->output_tensor_size_bytes);
    if (!output_tensor_plaintext) {
        fprintf(stderr, "CLIENT_APP_ONNX_ERROR: Failed to allocate memory for output tensor.\n");
        free(response_msg_ptr);
        return EXIT_FAILURE;
    }

    if (response_payload->masking_level != current_masking_level) {
         fprintf(stderr, "CLIENT_APP_ONNX_WARNING: Masking level mismatch between request (%d) and response (%d).\n",
                current_masking_level, response_payload->masking_level);
    }

    if (response_payload->masking_level == MASKING_AES_GCM) {
        printf("CLIENT_APP_ONNX_LOG: Decrypting output tensor (%u bytes) using AES-GCM...\n", response_payload->output_tensor_size_bytes);
        ret = libos_aes_gcm_decrypt(aes_key, response_payload->iv,
                                    response_payload->output_tensor, // Changed from masked_output_tensor
                                    response_payload->output_tensor_size_bytes,
                                    response_payload->tag,
                                    (unsigned char*)output_tensor_plaintext,
                                    NULL, 0); 
        if (ret != 0) {
            fprintf(stderr, "CLIENT_APP_ONNX_ERROR: Decryption of output tensor failed: %d\n", ret);
            free(output_tensor_plaintext);
            free(response_msg_ptr);
            return EXIT_FAILURE;
        }
    } else { // MASKING_NONE
        printf("CLIENT_APP_ONNX_LOG: Received output tensor as plaintext (%u bytes).\n", response_payload->output_tensor_size_bytes);
        memcpy(output_tensor_plaintext, response_payload->output_tensor, response_payload->output_tensor_size_bytes);
    }

    // 6. Verification (Print Top-K classes)
    uint32_t num_output_elements = response_payload->output_tensor_size_bytes / sizeof(float);
    printf("CLIENT_APP_ONNX_LOG: ONNX Inference successful. Output tensor has %u elements.\n", num_output_elements);
    if (num_output_elements == ONNX_MODEL_OUTPUT_CLASSES) { // Assuming MobileNetV2 output
        print_top_k_classes(output_tensor_plaintext, num_output_elements, 5);
    } else {
        printf("CLIENT_APP_ONNX_LOG: Output tensor has %u elements, not printing classes (expected %d for MobileNetV2).\n",
               num_output_elements, ONNX_MODEL_OUTPUT_CLASSES);
    }
    
    // 7. Cleanup
    if (input_tensor_plaintext) {
        if (current_masking_level == MASKING_NONE) {
            printf("CLIENT_APP_ONNX_LOG: Freeing page-locked (pinned) host memory for input_tensor_plaintext using cudaFreeHost...\n");
            CUDA_CHECK(cudaFreeHost(input_tensor_plaintext));
        } else { // MASKING_AES_GCM
            printf("CLIENT_APP_ONNX_LOG: Freeing host memory for input_tensor_plaintext using free()...\n");
            free(input_tensor_plaintext);
        }
        input_tensor_plaintext = NULL;
    }

    if (output_tensor_plaintext) {
        free(output_tensor_plaintext);
        output_tensor_plaintext = NULL;
    }
    if (response_msg_ptr) {
        free(response_msg_ptr); 
        response_msg_ptr = NULL;
    }
    printf("CLIENT_APP_ONNX_LOG: ONNX client application finished successfully.\n");

    return EXIT_SUCCESS;
}
