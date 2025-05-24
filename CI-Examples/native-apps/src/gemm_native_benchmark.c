#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Define matrix dimensions for this example
#define GEMM_M_NATIVE 512
#define GEMM_N_NATIVE 512
#define GEMM_K_NATIVE 512

// Helper to initialize a matrix with sample data
static void initialize_matrix_native(float* matrix, int rows, int cols, float val_offset) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = (float)(i + j + val_offset);
        }
    }
}

// Helper for cuBLAS errors
static const char* cublasGetErrorString(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "Unknown cuBLAS error";
    }
}

int main(int argc, char *argv[]) {
    int M = GEMM_M_NATIVE;
    int N = GEMM_N_NATIVE;
    int K = GEMM_K_NATIVE;

    if (argc == 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
        if (M <=0 || N <= 0 || K <= 0) {
            fprintf(stderr, "Invalid matrix dimensions. Using defaults.\n");
            M = GEMM_M_NATIVE; N = GEMM_N_NATIVE; K = GEMM_K_NATIVE;
        }
    }
    printf("NATIVE_GEMM_LOG: Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);

    float *h_A, *h_B, *h_C; // Host matrices
    float *d_A, *d_B, *d_C; // Device matrices
    cublasHandle_t handle;
    cublasStatus_t status;
    cudaError_t cuda_stat;
    cudaEvent_t start_event, stop_event;
    float gpu_time_ms = 0.0f;

    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C = (float*)malloc(size_C);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "NATIVE_GEMM_ERROR: Failed to allocate host memory.\n");
        return 1;
    }

    initialize_matrix_native(h_A, M, K, 1.0f);
    initialize_matrix_native(h_B, K, N, 2.0f);

    // Initialize cuBLAS
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "NATIVE_GEMM_ERROR: cublasCreate failed: %s\n", cublasGetErrorString(status));
        goto cleanup_host;
    }

    // Allocate GPU memory
    cuda_stat = cudaMalloc((void**)&d_A, size_A);
    if (cuda_stat != cudaSuccess) { fprintf(stderr, "cudaMalloc d_A failed: %s\n", cudaGetErrorString(cuda_stat)); goto cleanup_cublas; }
    cuda_stat = cudaMalloc((void**)&d_B, size_B);
    if (cuda_stat != cudaSuccess) { fprintf(stderr, "cudaMalloc d_B failed: %s\n", cudaGetErrorString(cuda_stat)); goto cleanup_gpu_A; }
    cuda_stat = cudaMalloc((void**)&d_C, size_C);
    if (cuda_stat != cudaSuccess) { fprintf(stderr, "cudaMalloc d_C failed: %s\n", cudaGetErrorString(cuda_stat)); goto cleanup_gpu_B; }

    // Copy matrices from host to device
    cuda_stat = cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    if (cuda_stat != cudaSuccess) { fprintf(stderr, "cudaMemcpy d_A failed: %s\n", cudaGetErrorString(cuda_stat)); goto cleanup_gpu_C; }
    cuda_stat = cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    if (cuda_stat != cudaSuccess) { fprintf(stderr, "cudaMemcpy d_B failed: %s\n", cudaGetErrorString(cuda_stat)); goto cleanup_gpu_C; }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    printf("NATIVE_GEMM_LOG: Performing cuBLAS SGEMM...\n");
    cudaEventRecord(start_event, 0);
    // cuBLAS expects column-major. For row-major A, B, C and C = A*B:
    // C_colmajor_transpose(N,M) = B_colmajor_transpose(N,K) * A_colmajor_transpose(K,M)
    // This means calling sgemm with (N, M, K) and providing B then A.
    // lda=M, ldb=K, ldc=N (if matrices were column major)
    // If using CUBLAS_OP_T, it expects data to be already in column-major for that op.
    // For A(MxK, row-major) and B(KxN, row-major), to get C(MxN, row-major)
    // C_transpose = B_transpose * A_transpose
    // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, M, K, &alpha, d_B(KxN), K, d_A(MxK), M, &beta, d_C(MxN), N)
    status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, 
                         N, M, K, 
                         &alpha, 
                         d_B, K, // B is KxN, leading dim for B (as if row-major) is N, but for B^T (NxK) it's K
                         d_A, M, // A is MxK, leading dim for A (as if row-major) is K, but for A^T (KxM) it's M
                         &beta, 
                         d_C, N); // C is MxN, leading dim for C (as if row-major) is N. Result C^T is NxM, ldc=N
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&gpu_time_ms, start_event, stop_event);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "NATIVE_GEMM_ERROR: cublasSgemm failed: %s\n", cublasGetErrorString(status));
        goto cleanup_gpu_C;
    }
    printf("NATIVE_GEMM_LOG: cuBLAS SGEMM successful.\n");
    printf("NATIVE_GEMM_LOG: GPU execution time (cublasSgemm): %.3f ms\n", gpu_time_ms);

    // Copy result back to host
    cuda_stat = cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    if (cuda_stat != cudaSuccess) {
        fprintf(stderr, "NATIVE_GEMM_ERROR: cudaMemcpy for d_C failed: %s\n", cudaGetErrorString(cuda_stat));
    } else {
        printf("NATIVE_GEMM_LOG: Result C copied back to host.\n");
        // Optional: Verification can be added here by comparing h_C with CPU computed result
    }

cleanup_gpu_C:
    cudaFree(d_C);
cleanup_gpu_B:
    cudaFree(d_B);
cleanup_gpu_A:
    cudaFree(d_A);
cleanup_cublas:
    cublasDestroy(handle);
cleanup_host:
    free(h_A); free(h_B); free(h_C);
    cudaEventDestroy(start_event); cudaEventDestroy(stop_event);

    printf("NATIVE_GEMM_LOG: Native GEMM benchmark finished.\n");
    return (status == CUBLAS_STATUS_SUCCESS && cuda_stat == cudaSuccess) ? 0 : 1;
}
