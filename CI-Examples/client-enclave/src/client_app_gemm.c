#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>    
#include <assert.h>
#include <math.h>    

// Gramine includes
#include "libos_ipc.h"
#include "libos_aes_gcm.h"

// Shared header with service
#include "shared_service.h" // From CI-Examples/common/

// Define matrix dimensions for this example
// M, N, K should be <= MAX_GEMM_DIM_SIZE (512)
#define GEMM_M 128
#define GEMM_N 128
#define GEMM_K 128

// Helper to initialize a matrix with sample data
static void initialize_matrix(float* matrix, int rows, int cols, float val_offset) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = (float)(i + j + val_offset);
        }
    }
}

// Simple CPU-based SGEMM for verification (C = A * B)
// A: M x K, B: K x N, C: M x N
static void sgemm_cpu(int M, int N, int K, const float* A, const float* B, float* C) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

// Helper to generate random IVs
static void generate_iv(unsigned char* iv, size_t iv_len) {
    for (size_t i = 0; i < iv_len; ++i) {
        iv[i] = (unsigned char)(rand() % 256);
    }
}

// Helper to compare matrices (first few elements and a corner element)
static int verify_matrices(int M, int N, const float* mat1, const float* mat2, float tolerance) {
    int mismatches = 0;
    printf("CLIENT_APP_GEMM: Verifying matrices (up to 5 elements and last):\n");
    for (int i = 0; i < M * N; ++i) {
        if (fabs(mat1[i] - mat2[i]) > tolerance) {
            mismatches++;
            if (mismatches < 5 || i == (M * N -1) ) { // Print first few and last mismatch
                fprintf(stderr, "  Mismatch at index %d: Expected %.5f, Got %.5f\n",
                        i, mat1[i], mat2[i]);
            }
        }
         if (i < 5 || i == (M*N-1)) { // Print first few and last element for visual check
             printf("  Idx %d: Expected %.2f, Got %.2f %s\n", i, mat1[i], mat2[i], (fabs(mat1[i] - mat2[i]) > tolerance ? "<- MISMATCH" : ""));
         }
    }
    return mismatches;
}


int main(int argc, char *argv[]) {
    printf("CLIENT_APP_GEMM_LOG: Starting GEMM client application...\n");
    clock_t start_time, end_time;
    double cpu_time_used;

    gpu_data_masking_level_t current_masking_level = MASKING_AES_GCM; // Default
    // Basic argument parsing for --masking
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--masking") == 0 && (i + 1 < argc)) {
            if (strcmp(argv[i+1], "none") == 0) {
                current_masking_level = MASKING_NONE;
            } else if (strcmp(argv[i+1], "aes_gcm") == 0) {
                current_masking_level = MASKING_AES_GCM;
            } else {
                fprintf(stderr, "CLIENT_APP_GEMM_WARNING: Unknown masking mode '%s'. Defaulting to AES-GCM.\n", argv[i+1]);
            }
            i++; // Skip next argument
        }
    }
    printf("CLIENT_APP_GEMM_LOG: Using masking mode: %s\n",
           current_masking_level == MASKING_AES_GCM ? "AES-GCM" : "None");

    unsigned char aes_key[GCM_KEY_SIZE_BYTES];
    if (current_masking_level == MASKING_AES_GCM) {
        for(int i=0; i < GCM_KEY_SIZE_BYTES; ++i) aes_key[i] = (unsigned char)(i + 0xBB); // Example key
    }
    srand(time(NULL)); 

    // 1. Initialize input matrices
    const int M = GEMM_M;
    const int N = GEMM_N;
    const int K = GEMM_K;

    if (M * K > MAX_GEMM_MATRIX_ELEMENTS || K * N > MAX_GEMM_MATRIX_ELEMENTS || M * N > MAX_GEMM_MATRIX_ELEMENTS) {
        fprintf(stderr, "CLIENT_APP_GEMM_ERROR: Matrix dimensions exceed MAX_GEMM_MATRIX_ELEMENTS.\n");
        return EXIT_FAILURE;
    }

    float* h_A = (float*)malloc(M * K * sizeof(float));
    float* h_B = (float*)malloc(K * N * sizeof(float));
    float* h_C_expected = (float*)malloc(M * N * sizeof(float));
    float* h_C_result_decrypted = (float*)malloc(M * N * sizeof(float));

    if (!h_A || !h_B || !h_C_expected || !h_C_result_decrypted) {
        fprintf(stderr, "CLIENT_APP_GEMM_ERROR: Failed to allocate memory for host matrices.\n");
        return EXIT_FAILURE;
    }

    initialize_matrix(h_A, M, K, 1.0f);
    initialize_matrix(h_B, K, N, 2.0f);

    printf("CLIENT_APP_GEMM_LOG: Performing local CPU SGEMM for verification...\n");
    sgemm_cpu(M, N, K, h_A, h_B, h_C_expected);

    // 2. Prepare request payload
    gemm_request_payload_t request_payload;
    memset(&request_payload, 0, sizeof(request_payload));
    request_payload.M = M;
    request_payload.N = N;
    request_payload.K = K;
    request_payload.matrix_a_size_bytes = M * K * sizeof(float);
    request_payload.matrix_b_size_bytes = K * N * sizeof(float);
    request_payload.masking_level = current_masking_level;
    int ret;

    assert(request_payload.matrix_a_size_bytes <= MAX_GEMM_MATRIX_SIZE_BYTES);
    assert(request_payload.matrix_b_size_bytes <= MAX_GEMM_MATRIX_SIZE_BYTES);

    if (current_masking_level == MASKING_AES_GCM) {
        printf("CLIENT_APP_GEMM_LOG: Encrypting input matrices A and B using AES-GCM...\n");
        generate_iv(request_payload.iv_a, GCM_IV_SIZE_BYTES);
        ret = libos_aes_gcm_encrypt(aes_key, request_payload.iv_a, (const unsigned char*)h_A, 
                                        request_payload.matrix_a_size_bytes,
                                        request_payload.matrix_a, request_payload.tag_a, NULL, 0); // Changed to matrix_a
        if (ret != 0) { fprintf(stderr, "CLIENT_APP_GEMM_ERROR: Encryption of matrix A failed: %d\n", ret); goto cleanup_exit; }

        generate_iv(request_payload.iv_b, GCM_IV_SIZE_BYTES);
        ret = libos_aes_gcm_encrypt(aes_key, request_payload.iv_b, (const unsigned char*)h_B, 
                                        request_payload.matrix_b_size_bytes,
                                        request_payload.matrix_b, request_payload.tag_b, NULL, 0); // Changed to matrix_b
        if (ret != 0) { fprintf(stderr, "CLIENT_APP_GEMM_ERROR: Encryption of matrix B failed: %d\n", ret); goto cleanup_exit; }
    } else { // MASKING_NONE
        printf("CLIENT_APP_GEMM_LOG: Preparing input matrices A and B as plaintext...\n");
        memcpy(request_payload.matrix_a, h_A, request_payload.matrix_a_size_bytes);
        memcpy(request_payload.matrix_b, h_B, request_payload.matrix_b_size_bytes);
    }

    // 3. Prepare IPC message
    struct libos_ipc_msg request_msg;
    init_ipc_msg(&request_msg, GEMM_REQUEST, 
                 sizeof(libos_ipc_msg_header_t) + sizeof(gemm_request_payload_t));
    if (sizeof(gemm_request_payload_t) > sizeof(request_msg.data)) {
         fprintf(stderr, "CLIENT_APP_GEMM_ERROR: gemm_request_payload_t size (%zu) > libos_ipc_msg.data size (%zu)\n",
                 sizeof(gemm_request_payload_t), sizeof(request_msg.data));
         goto cleanup_exit;
    }
    memcpy(request_msg.data, &request_payload, sizeof(gemm_request_payload_t));

    // 4. IPC Communication
    IDTYPE shared_enclave_vmid = 1; 
    struct libos_ipc_msg* response_msg_ptr = NULL;
    printf("CLIENT_APP_GEMM_LOG: Sending GEMM request to shared enclave (VMID: %u)...\n", shared_enclave_vmid);
    start_time = clock();
    ret = ipc_send_msg_and_get_response(shared_enclave_vmid, &request_msg, (void**)&response_msg_ptr);
    end_time = clock();
    cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;

    if (ret < 0) { fprintf(stderr, "CLIENT_APP_GEMM_ERROR: IPC failed: %s\n", unix_strerror(ret)); goto cleanup_exit_resp; }
    if (!response_msg_ptr) { fprintf(stderr, "CLIENT_APP_GEMM_ERROR: Received NULL response_msg_ptr.\n"); goto cleanup_exit; }
    printf("CLIENT_APP_GEMM_LOG: Received response. End-to-end time: %.4f seconds\n", cpu_time_used);

    // 5. Handle Response
    if (GET_UNALIGNED(response_msg_ptr->header.code) != GEMM_REQUEST) { // Or specific GEMM_RESPONSE code
        fprintf(stderr, "CLIENT_APP_GEMM_ERROR: Unexpected message code: %u\n", GET_UNALIGNED(response_msg_ptr->header.code));
        goto cleanup_exit_resp;
    }
    size_t expected_resp_payload_size = sizeof(gemm_response_payload_t);
    if (GET_UNALIGNED(response_msg_ptr->header.size) != (sizeof(libos_ipc_msg_header_t) + expected_resp_payload_size)) {
         fprintf(stderr, "CLIENT_APP_GEMM_ERROR: Unexpected response size: %lu (expected payload %zu)\n",
                 GET_UNALIGNED(response_msg_ptr->header.size) - sizeof(libos_ipc_msg_header_t), expected_resp_payload_size);
        goto cleanup_exit_resp;
    }
    gemm_response_payload_t* response_payload = (gemm_response_payload_t*)response_msg_ptr->data;
    if (response_payload->status != 0) {
        fprintf(stderr, "CLIENT_APP_GEMM_ERROR: Shared enclave reported GEMM error: %d\n", response_payload->status);
        goto cleanup_exit_resp;
    }
    if (response_payload->matrix_c_size_bytes != (M * N * sizeof(float))) {
        fprintf(stderr, "CLIENT_APP_GEMM_ERROR: Response matrix C size mismatch. Expected %zu, Got %u\n",
                M * N * sizeof(float), response_payload->matrix_c_size_bytes);
        goto cleanup_exit_resp;
    }

    if (response_payload->masking_level != current_masking_level) {
        fprintf(stderr, "CLIENT_APP_GEMM_WARNING: Masking level mismatch between request (%d) and response (%d).\n",
                current_masking_level, response_payload->masking_level);
    }

    if (response_payload->masking_level == MASKING_AES_GCM) {
        printf("CLIENT_APP_GEMM_LOG: Decrypting result matrix C (AES-GCM)...\n");
        ret = libos_aes_gcm_decrypt(aes_key, response_payload->iv_c, 
                                    response_payload->matrix_c, // Changed from masked_matrix_c
                                    response_payload->matrix_c_size_bytes,
                                    response_payload->tag_c, (unsigned char*)h_C_result_decrypted, NULL, 0);
        if (ret != 0) { fprintf(stderr, "CLIENT_APP_GEMM_ERROR: Decryption of matrix C failed: %d\n", ret); goto cleanup_exit_resp; }
    } else { // MASKING_NONE
        printf("CLIENT_APP_GEMM_LOG: Received result matrix C as plaintext.\n");
        memcpy(h_C_result_decrypted, response_payload->matrix_c, response_payload->matrix_c_size_bytes);
    }

    // 6. Verify Result
    printf("CLIENT_APP_GEMM_LOG: Verifying results...\n");
    int mismatches = verify_matrices(M, N, h_C_expected, h_C_result_decrypted, 1e-3f); // Tolerance for SGEMM

    if (mismatches == 0) {
        printf("CLIENT_APP_GEMM_SUCCESS: SGEMM results verified successfully!\n");
    } else {
        fprintf(stderr, "CLIENT_APP_GEMM_FAILURE: %d mismatches found in SGEMM results.\n", mismatches);
        goto cleanup_exit_resp;
    }

    // 7. Cleanup
    free(response_msg_ptr);
    free(h_A); free(h_B); free(h_C_expected); free(h_C_result_decrypted);
    printf("CLIENT_APP_GEMM_LOG: Client application finished successfully.\n");
    return EXIT_SUCCESS;

cleanup_exit_resp:
    if(response_msg_ptr) free(response_msg_ptr);
cleanup_exit:
    free(h_A); free(h_B); free(h_C_expected); free(h_C_result_decrypted);
    return EXIT_FAILURE;
}
