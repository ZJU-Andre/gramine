// SPDX-License-Identifier: GPL-3.0-only OR BSD-3-Clause
/*
 * Copyright (C) 2023 Gramine contributors
 *
 * This file is part of the Gramine project.
 */

#ifndef LIBOS_GPU_H
#define LIBOS_GPU_H

#include <stdint.h>
#include <stddef.h>

typedef void* gramine_device_ptr_t;

/* Initialization and shutdown */
int gramine_cuda_init(void);
int gramine_cuda_shutdown(void);

/* Memory management */
int gramine_cuda_malloc_device(gramine_device_ptr_t* device_ptr, size_t size);
int gramine_cuda_memcpy_host_to_device(gramine_device_ptr_t device_ptr, const void* host_ptr, size_t size);
int gramine_cuda_memcpy_device_to_host(void* host_ptr, gramine_device_ptr_t device_ptr, size_t size);
int gramine_cuda_free_device(gramine_device_ptr_t device_ptr);

/* Kernel launch */
int gramine_cuda_launch_kernel_by_name(const char* kernel_name, void** kernel_args,
                                       unsigned int grid_dim_x, unsigned int grid_dim_y, unsigned int grid_dim_z,
                                       unsigned int block_dim_x, unsigned int block_dim_y, unsigned int block_dim_z,
                                       unsigned int shared_mem_bytes);

#endif /* LIBOS_GPU_H */
