#ifndef VECTOR_ADD_H
#define VECTOR_ADD_H

#ifdef __cplusplus
extern "C" {
#endif

int launch_vector_add_cuda(float* h_A_out, 
                           const float* h_B_in, 
                           const float* h_C_in, 
                           int n,
                           int* cuda_error_code,       // Output CUDA error code
                           const char** cuda_error_str); // Output CUDA error string pointer
#ifdef __cplusplus
}
#endif

#endif // VECTOR_ADD_H
