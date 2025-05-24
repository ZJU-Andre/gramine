#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>    // For srand, clock
#include <assert.h>
#include <math.h>    // For fabs

// Gramine includes
#include "libos_ipc.h"
#include "libos_aes_gcm.h"

// Shared header with service
#include "shared_service.h" // From CI-Examples/common/

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

    // TODO: Secure Key and IV Management. For this example, they are hardcoded.
    unsigned char aes_key[GCM_KEY_SIZE_BYTES];
    for(int i=0; i < GCM_KEY_SIZE_BYTES; ++i) aes_key[i] = (unsigned char)(i + 0xAA); // Example key

    srand(time(NULL)); 

    // 1. Initialize input tensor data (dummy data for MobileNetV2 1x3x224x224)
    const uint32_t input_elements = ONNX_MODEL_INPUT_CHANNELS * ONNX_MODEL_INPUT_HEIGHT * ONNX_MODEL_INPUT_WIDTH;
    const uint32_t input_tensor_size_bytes = input_elements * sizeof(float);
    assert(input_tensor_size_bytes <= MAX_ONNX_INPUT_SIZE_BYTES);

    float* input_tensor_plaintext = (float*)malloc(input_tensor_size_bytes);
    if (!input_tensor_plaintext) {
        fprintf(stderr, "CLIENT_APP_ONNX_ERROR: Failed to allocate memory for input tensor.\n");
        return EXIT_FAILURE;
    }
    generate_random_float_tensor(input_tensor_plaintext, input_elements);
    printf("CLIENT_APP_ONNX_LOG: Generated dummy input tensor of size %u bytes (%u elements).\n",
           input_tensor_size_bytes, input_elements);

    // 2. Prepare request payload
    onnx_inference_request_payload_t request_payload;
    memset(&request_payload, 0, sizeof(request_payload));
    request_payload.input_tensor_size_bytes = input_tensor_size_bytes;
    // Optional: Populate model_name if service uses it.
    // strncpy(request_payload.model_name, "mobilenetv2-7.onnx", MAX_MODEL_NAME_LEN -1);
    // request_payload.model_name_len = strlen(request_payload.model_name);

    printf("CLIENT_APP_ONNX_LOG: Encrypting input tensor...\n");
    generate_iv(request_payload.iv, GCM_IV_SIZE_BYTES);
    int ret = libos_aes_gcm_encrypt(aes_key, request_payload.iv,
                                    (const unsigned char*)input_tensor_plaintext,
                                    input_tensor_size_bytes,
                                    request_payload.masked_input_tensor,
                                    request_payload.tag,
                                    NULL, 0); // No AAD
    if (ret != 0) {
        fprintf(stderr, "CLIENT_APP_ONNX_ERROR: Encryption of input tensor failed: %d\n", ret);
        free(input_tensor_plaintext);
        return EXIT_FAILURE;
    }
    free(input_tensor_plaintext); // Plaintext no longer needed by client after encryption

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

    printf("CLIENT_APP_ONNX_LOG: Decrypting output tensor (%u bytes)...\n", response_payload->output_tensor_size_bytes);
    ret = libos_aes_gcm_decrypt(aes_key, response_payload->iv,
                                response_payload->masked_output_tensor,
                                response_payload->output_tensor_size_bytes,
                                response_payload->tag,
                                (unsigned char*)output_tensor_plaintext,
                                NULL, 0); // No AAD
    if (ret != 0) {
        fprintf(stderr, "CLIENT_APP_ONNX_ERROR: Decryption of output tensor failed: %d\n", ret);
        free(output_tensor_plaintext);
        free(response_msg_ptr);
        return EXIT_FAILURE;
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
    free(output_tensor_plaintext);
    free(response_msg_ptr); 
    printf("CLIENT_APP_ONNX_LOG: ONNX client application finished successfully.\n");

    return EXIT_SUCCESS;
}
