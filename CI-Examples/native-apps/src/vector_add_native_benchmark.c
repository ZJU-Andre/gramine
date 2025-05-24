#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h> // For cudaEvent_t

// Assume vector_add.h is accessible (e.g., copied or path provided in Makefile)
#include "vector_add.h" 

#define DEFAULT_N_ELEMENTS (1024 * 1024) // 2^20 elements

// Helper to print float arrays for debugging (optional)
static void print_float_array_sample(const char* label, const float* data, int n, int sample_count) {
    printf("%s (sample of %d elements):\n", label, sample_count);
    for (int i = 0; i < sample_count && i < n; ++i) {
        printf("%.2f ", data[i]);
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    int n = DEFAULT_N_ELEMENTS;
    if (argc > 1) {
        n = atoi(argv[1]);
        if (n <= 0) {
            fprintf(stderr, "Invalid vector size: %s. Using default %d.\n", argv[1], DEFAULT_N_ELEMENTS);
            n = DEFAULT_N_ELEMENTS;
        }
    }

    printf("NATIVE_VECTOR_ADD_LOG: Vector size: %d elements\n", n);

    float* h_A = (float*)malloc(n * sizeof(float));
    float* h_B = (float*)malloc(n * sizeof(float));
    float* h_C = (float*)malloc(n * sizeof(float));

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "NATIVE_VECTOR_ADD_ERROR: Failed to allocate host memory.\n");
        return 1;
    }

    for (int i = 0; i < n; i++) {
        h_B[i] = (float)i;
        h_C[i] = (float)i * 2.0f;
    }

    cudaEvent_t start_event, stop_event;
    float gpu_time_ms = 0.0f;
    int cuda_error_code = 0;
    const char* cuda_error_str = NULL;

    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    printf("NATIVE_VECTOR_ADD_LOG: Launching CUDA kernel...\n");
    
    cudaEventRecord(start_event, 0);
    int ret = launch_vector_add_cuda(h_A, h_B, h_C, n, &cuda_error_code, &cuda_error_str);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&gpu_time_ms, start_event, stop_event);

    if (ret != 0) {
        fprintf(stderr, "NATIVE_VECTOR_ADD_ERROR: CUDA execution failed. Code: %d, Str: %s\n", 
                cuda_error_code, cuda_error_str ? cuda_error_str : "N/A");
        free(h_A); free(h_B); free(h_C);
        cudaEventDestroy(start_event); cudaEventDestroy(stop_event);
        return 1;
    }

    printf("NATIVE_VECTOR_ADD_LOG: CUDA kernel execution successful.\n");
    printf("NATIVE_VECTOR_ADD_LOG: GPU execution time: %.3f ms\n", gpu_time_ms);

    // Verification (optional, sample check)
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (fabs(h_A[i] - (h_B[i] + h_C[i])) > 1e-5) {
            errors++;
        }
    }
    if (errors > 0) {
        printf("NATIVE_VECTOR_ADD_LOG: Verification FAILED with %d errors.\n", errors);
    } else {
        printf("NATIVE_VECTOR_ADD_LOG: Verification PASSED.\n");
    }
    // print_float_array_sample("Result A (Native)", h_A, n, 10);

    free(h_A);
    free(h_B);
    free(h_C);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    printf("NATIVE_VECTOR_ADD_LOG: Native vector add benchmark finished.\n");
    return 0;
}
