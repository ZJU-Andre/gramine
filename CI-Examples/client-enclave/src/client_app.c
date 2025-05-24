#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>    // For srand
#include <assert.h>
#include <math.h>    // For fabs

// Gramine includes - assumed to be in include path
#include "libos_ipc.h"
#include "libos_aes_gcm.h"

// Shared header with service
#include "shared_service.h" // From CI-Examples/common/

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
    printf("CLIENT_APP_LOG: Starting client application...\n");

    // TODO: Secure Key and IV Management. For this example, they are hardcoded.
    // In a real application, keys should be protected (e.g., sealed by SGX or derived via attestation)
    // and IVs should be generated uniquely for each encryption using a CSPRNG.
    unsigned char aes_key[GCM_KEY_SIZE_BYTES];
    for(int i=0; i < GCM_KEY_SIZE_BYTES; ++i) aes_key[i] = (unsigned char)(i + 0x55); // Example key

    srand(time(NULL)); // Seed for random data generation

    // 1. Initialize input data
    const int num_elements = VECTOR_ARRAY_DEFAULT_ELEMENTS;
    float h_B[num_elements];
    float h_C[num_elements];
    float h_A_expected[num_elements];
    float h_A_result[num_elements];

    generate_float_array(h_B, num_elements);
    generate_float_array(h_C, num_elements);

    for (int i = 0; i < num_elements; ++i) {
        h_A_expected[i] = h_B[i] + h_C[i];
    }
    // print_float_array("Host B (plaintext)", h_B, num_elements);
    // print_float_array("Host C (plaintext)", h_C, num_elements);
    // print_float_array("Host A_expected (plaintext)", h_A_expected, num_elements);


    // 2. Prepare request payload
    vector_add_request_payload_t request_payload;
    memset(&request_payload, 0, sizeof(request_payload));
    request_payload.array_len_elements = num_elements;

    printf("CLIENT_APP_LOG: Encrypting input arrays B and C...\n");

    // Encrypt B
    generate_iv(request_payload.iv_b, GCM_IV_SIZE_BYTES);
    int ret = libos_aes_gcm_encrypt(aes_key, request_payload.iv_b,
                                    (const unsigned char*)h_B, num_elements * sizeof(float),
                                    request_payload.masked_data_b, request_payload.tag_b,
                                    NULL, 0); // No AAD for this example
    if (ret != 0) {
        fprintf(stderr, "CLIENT_APP_ERROR: Encryption of array B failed: %d\n", ret);
        return EXIT_FAILURE;
    }

    // Encrypt C
    generate_iv(request_payload.iv_c, GCM_IV_SIZE_BYTES);
    ret = libos_aes_gcm_encrypt(aes_key, request_payload.iv_c,
                                (const unsigned char*)h_C, num_elements * sizeof(float),
                                request_payload.masked_data_c, request_payload.tag_c,
                                NULL, 0); // No AAD for this example
    if (ret != 0) {
        fprintf(stderr, "CLIENT_APP_ERROR: Encryption of array C failed: %d\n", ret);
        return EXIT_FAILURE;
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
        free(response_msg_ptr);
        return EXIT_FAILURE;
    }
    // Expected size: header + specific payload
    size_t expected_resp_payload_size = sizeof(vector_add_response_payload_t);
    if (GET_UNALIGNED(response_msg_ptr->header.size) != (sizeof(libos_ipc_msg_header_t) + expected_resp_payload_size)) {
         fprintf(stderr, "CLIENT_APP_ERROR: Received message with unexpected size: %lu (expected %zu for payload)\n",
                 GET_UNALIGNED(response_msg_ptr->header.size), expected_resp_payload_size);
        free(response_msg_ptr);
        return EXIT_FAILURE;
    }

    vector_add_response_payload_t* response_payload = (vector_add_response_payload_t*)response_msg_ptr->data;

    if (response_payload->status != 0) {
        fprintf(stderr, "CLIENT_APP_ERROR: Shared enclave reported error: %d\n", response_payload->status);
        // If CUDA error, it might be in status. How to get string? Service should provide it.
        // For now, just the code.
        free(response_msg_ptr);
        return EXIT_FAILURE;
    }
    if (response_payload->array_len_elements != num_elements) {
        fprintf(stderr, "CLIENT_APP_ERROR: Response array length mismatch. Expected %d, Got %u\n",
                num_elements, response_payload->array_len_elements);
        free(response_msg_ptr);
        return EXIT_FAILURE;
    }

    printf("CLIENT_APP_LOG: Decrypting result array A...\n");
    ret = libos_aes_gcm_decrypt(aes_key, response_payload->iv_a,
                                response_payload->masked_data_a, num_elements * sizeof(float),
                                response_payload->tag_a, (unsigned char*)h_A_result,
                                NULL, 0); // No AAD for this example
    if (ret != 0) {
        fprintf(stderr, "CLIENT_APP_ERROR: Decryption of result array A failed: %d\n", ret);
        free(response_msg_ptr);
        return EXIT_FAILURE;
    }
    // print_float_array("Host A_result (decrypted)", h_A_result, num_elements);

    // 6. Verify Result
    printf("CLIENT_APP_LOG: Verifying results...\n");
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
        free(response_msg_ptr);
        return EXIT_FAILURE;
    }

    // 7. Cleanup
    free(response_msg_ptr); // ipc_send_msg_and_get_response allocates this
    printf("CLIENT_APP_LOG: Client application finished successfully.\n");

    return EXIT_SUCCESS;
}
