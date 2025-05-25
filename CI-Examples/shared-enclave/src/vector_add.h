#ifndef VECTOR_ADD_H
#define VECTOR_ADD_H

#include <stdint.h> // For uint64_t
#include <stdbool.h> // For bool type

#ifdef __cplusplus
extern "C" {
#endif

int launch_vector_add_cuda(
    bool use_dma_for_output_a,       // New: Flag to indicate if DMA output to client host is used
    uint64_t dest_client_host_ptr_a, // New: Client-provided host pointer for output A (if DMA)
    float* h_A_out_fallback,         // Existing h_A_out, now a fallback if DMA output is not used
    const float* h_B_in, 
    const float* h_C_in, 
    int n,
    int* cuda_error_code,
    const char** cuda_error_str,
    bool use_dma_for_b,
    uint64_t d_ptr_b_client,
    bool use_dma_for_c,
    uint64_t d_ptr_c_client
);
#ifdef __cplusplus
}
#endif

#endif // VECTOR_ADD_H
