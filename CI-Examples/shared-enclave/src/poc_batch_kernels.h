#ifndef POC_BATCH_KERNELS_H
#define POC_BATCH_KERNELS_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Copies data from one device buffer to another asynchronously.
 * @param output_device_ptr Pointer to the output buffer on the GPU.
 * @param input_device_ptr Pointer to the input buffer on the GPU.
 * @param size_bytes Number of bytes to copy.
 * @param stream CUDA stream to execute on.
 * @return cudaError_t status.
 */
cudaError_t launch_poc_copy_kernel_dto_d(void* output_device_ptr, 
                                         const void* input_device_ptr, 
                                         uint32_t size_bytes, 
                                         cudaStream_t stream);

/**
 * @brief Fills a device buffer with a specific byte value asynchronously.
 * @param device_ptr Pointer to the device buffer on the GPU.
 * @param size_bytes Number of bytes to fill.
 * @param pattern_value Byte value to fill with.
 * @param stream CUDA stream to execute on.
 * @return cudaError_t status.
 */
cudaError_t launch_poc_generate_data_kernel(void* device_ptr, 
                                            uint32_t size_bytes, 
                                            unsigned char pattern_value, 
                                            cudaStream_t stream);

/**
 * @brief Multiplies elements of a float array on the device by a scale factor, out-of-place, asynchronously.
 * @param output_device_ptr Pointer to the output float array on the GPU.
 * @param input_device_ptr Pointer to the input float array on the GPU.
 * @param num_elements Number of float elements in the arrays.
 * @param scale_factor The float scale factor.
 * @param stream CUDA stream to execute on.
 * @return cudaError_t status.
 */
cudaError_t launch_poc_transform_kernel_float(float* output_device_ptr, 
                                              const float* input_device_ptr, 
                                              uint32_t num_elements, 
                                              float scale_factor, 
                                              cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // POC_BATCH_KERNELS_H
