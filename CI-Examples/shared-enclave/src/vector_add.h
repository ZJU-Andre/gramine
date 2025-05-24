#ifndef VECTOR_ADD_H
#define VECTOR_ADD_H

#include <stdint.h> // For uint64_t
#include <stdbool.h> // For bool type

#ifdef __cplusplus
extern "C" {
#endif

int launch_vector_add_cuda(float* h_A_out, 
                           const float* h_B_in, // Host pointer, used if not DMA for B
                           const float* h_C_in, // Host pointer, used if not DMA for C
                           int n,
                           int* cuda_error_code,       // Output CUDA error code
                           const char** cuda_error_str, // Output CUDA error string pointer
                           bool use_dma_for_b,
                           uint64_t d_ptr_b_client,
                           bool use_dma_for_c,
                           uint64_t d_ptr_c_client);
#ifdef __cplusplus
}
#endif

#endif // VECTOR_ADD_H
