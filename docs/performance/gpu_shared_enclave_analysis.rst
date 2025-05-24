.. SPDX-License-Identifier: LGPL-3.0-or-later
.. Copyright (C) 2023 Intel Corporation

##################################################
Performance Analysis of Shared Enclave GPU Architecture
##################################################

.. contents::
   :local:
   :depth: 2

1. Introduction
===============

Purpose
-------
The purpose of this document is to analyze the performance characteristics and overheads associated with the shared enclave GPU architecture in Gramine. This architecture is designed to allow multiple client enclaves to securely utilize a GPU managed by a central shared service enclave.

Architecture Overview
---------------------
The architecture under test involves:
*   **Client Enclaves:** Applications running in Gramine that require GPU-accelerated computations.
*   **Shared Enclave:** A dedicated Gramine enclave that has direct access to the host GPU devices (e.g., ``/dev/nvidia*``). It receives computation requests from client enclaves.
*   **Inter-Process Communication (IPC):** Client enclaves communicate with the shared enclave using Gramine's IPC mechanism (based on PAL pipes).
*   **Data Masking (AES-GCM):** Sensitive data transferred between client enclaves and the shared enclave, and data processed by the GPU, is protected using AES-GCM encryption and authentication. The shared enclave is responsible for unmasking data before GPU operations and re-masking results.

This analysis aims to quantify the performance impact of these components, particularly the IPC and data masking overheads, relative to native execution and a non-SGX Gramine (direct mode) execution.

2. Benchmarking Methodology
===========================

Test Applications
-----------------
The performance evaluation uses the following applications, each representing a different type of GPU workload:

*   **Vector Addition:** A memory-bound CUDA kernel that adds two large vectors.
    *   Workload: Varying vector sizes (e.g., 2^20, 2^22, 2^24 float elements).
*   **ONNX Model Inference (MobileNetV2):** A common computer vision model, representing a mixed CPU/GPU workload with moderate data sizes.
    *   Workload: Inference on a pre-defined input tensor (e.g., 1x3x224x224 floats).
*   **cuBLAS GEMM (SGEMM):** A compute-intensive General Matrix Multiplication operation using the cuBLAS library.
    *   Workload: Varying square matrix dimensions (e.g., 512x512, 1024x1024, 2048x2048 floats).

Execution Modes
---------------
Each benchmark application is run in the following modes:

1.  **Native Linux:** The application is compiled and run directly on the host Linux system, without Gramine. This serves as the baseline.
2.  **Gramine-direct:** The application is run within Gramine but without SGX protection. This helps isolate the overhead of the Gramine LibOS and PAL layers.
3.  **Gramine SGX (Shared Enclave Model):**
    *   The GPU-specific code (CUDA kernel, ONNX Runtime with CUDA EP, cuBLAS calls) is executed within the Shared Enclave.
    *   Client applications (running in separate Gramine SGX enclaves) send requests to the Shared Enclave via IPC. Data transferred is masked/unmasked using AES-GCM.

Metrics Collected
-----------------
The following performance metrics are collected:

*   **End-to-End Client Application Time:** For all modes, this is the total wall-clock time taken by the client application (or the native benchmark application) to complete its task. This is measured externally using ``/usr/bin/time -v``.
*   **Internal GPU Execution Time:** For operations directly using CUDA (Vector Addition, GEMM via cuBLAS) and ONNX inference on GPU, internal timing using ``cudaEventElapsedTime()`` is recorded within the application (native or shared enclave) to measure the GPU computation time specifically.
*   **SGX Mode Time Breakdown (Conceptual):** For the Gramine SGX mode, we aim to understand the contribution of different components. This typically requires instrumenting the client and server applications to measure:
    *   Client: Data preparation and AES-GCM encryption (masking).
    *   Client: Time spent in ``ipc_send_msg_and_get_response()`` (IPC send + wait for response).
    *   Client: Data unmasking (AES-GCM decryption) and result verification.
    *   Shared Enclave: Time spent in IPC receive logic and AES-GCM decryption (unmasking).
    *   Shared Enclave: GPU execution time (measured internally as described above).
    *   Shared Enclave: AES-GCM encryption (masking) of results and IPC send logic.
*   **System-Level Statistics:** Output from ``/usr/bin/time -v`` provides CPU time (user and system), maximum resident set size, page faults, and context switches.

Hardware/Software Environment
-----------------------------
Performance results are highly dependent on the underlying hardware and software stack. Users should document their environment when reproducing these benchmarks.

*   **CPU Model:** ``[User to fill, e.g., Intel Xeon E-2388G @ 3.20GHz]``
*   **GPU Model:** ``[User to fill, e.g., NVIDIA A100-SXM4-40GB]``
*   **RAM:** ``[User to fill, e.g., 128 GB]``
*   **OS Version:** ``[User to fill, e.g., Ubuntu 20.04.5 LTS]``
*   **NVIDIA Driver Version:** ``[User to fill, e.g., 510.47.03]``
*   **CUDA Toolkit Version:** ``[User to fill, e.g., 11.6]``
*   **ONNX Runtime Version (if used):** ``[User to fill, e.g., 1.15.1]``
*   **cuBLAS Version (if used):** ``[User to fill, e.g., part of CUDA 11.6]``
*   **Gramine Version:** ``[User to fill, e.g., v1.7]``
*   **SGX Driver/PSW Version:** ``[User to fill]``

Benchmark Execution
-------------------
*   **Number of Runs:** Each benchmark configuration should be run multiple times (e.g., ``NUM_RUNS = 5`` as defined in ``run_benchmarks.sh``).
*   **Averaging:** The reported end-to-end times should ideally be an average of these runs, excluding outliers if necessary (e.g., by taking the median or average of the middle three runs). Internal GPU execution times are typically averaged by the benchmark applications themselves over many iterations.
*   **Warm-up:** Consider including warm-up runs, especially for GPU operations, which are not included in the averaged results.

3. Performance Results (Hypothetical Discussion & Templates)
============================================================

This section presents templates for reporting performance results and discusses hypothetical outcomes and expectations. **Actual benchmark data needs to be collected by running the scripts.**

Vector Addition
---------------

**Table 1: Vector Addition - End-to-End Execution Time (seconds)**

+----------------------+----------+-----------------+---------------------------+
| Workload (Elements)  | Native   | Gramine-direct  | Gramine SGX (Shared Enc.) |
+======================+==========+=================+===========================+
| 2^20 (approx 1M)     | [time_n1]| [time_gd1]      | [time_sgx1]               |
+----------------------+----------+-----------------+---------------------------+
| 2^22 (approx 4M)     | [time_n2]| [time_gd2]      | [time_sgx2]               |
+----------------------+----------+-----------------+---------------------------+
| 2^24 (approx 16M)    | [time_n3]| [time_gd3]      | [time_sgx3]               |
+----------------------+----------+-----------------+---------------------------+

**Table 2: Vector Addition - Calculated Overheads vs Native**

+----------------------+---------------------------+-----------------------------------+
| Workload (Elements)  | Gramine-direct Overhead % | Gramine SGX (Shared Enc.) Ovhd. % |
+======================+===========================+===================================+
| 2^20                 | [ovhd_gd1]%               | [ovhd_sgx1]%                      |
+----------------------+---------------------------+-----------------------------------+
| 2^22                 | [ovhd_gd2]%               | [ovhd_sgx2]%                      |
+----------------------+---------------------------+-----------------------------------+
| 2^24                 | [ovhd_gd3]%               | [ovhd_sgx3]%                      |
+----------------------+---------------------------+-----------------------------------+

**Table 3: Vector Addition @ 2^22 Elements - SGX Mode Time Breakdown (ms - Hypothetical)**

+---------------------------------+------------+
| Component                       | Time (ms)  |
+=================================+============+
| Client: Data Prep & Masking     | [time_c_prep_mask_va] |
+---------------------------------+------------+
| Client: IPC Send + Wait         | [time_c_ipc_va]       |
+---------------------------------+------------+
| Client: Data Unmask & Verify  | [time_c_unmask_ver_va] |
+---------------------------------+------------+
| Shared Enc: IPC Recv & Unmask | [time_s_ipc_unmask_va] |
+---------------------------------+------------+
| Shared Enc: GPU Execution (CUDA)| [time_s_gpu_va]       |
+---------------------------------+------------+
| Shared Enc: Data Mask & IPC Send| [time_s_mask_ipc_va]  |
+---------------------------------+------------+
| **Client End-to-End Total**     | [time_c_total_va]     |
+---------------------------------+------------+

**Hypothetical Discussion Points (Vector Addition):**
    *   Vector addition is typically memory-bandwidth bound on the GPU. The actual kernel execution time might be relatively small, especially for smaller vector sizes.
    *   **Gramine-direct overhead:** Expected to be low, primarily due to PAL syscall translations and LibOS environment setup.
    *   **Gramine SGX overhead:** Expected to be significant. For vector addition, the overhead from data masking (AES-GCM encryption/decryption for two input vectors and one output vector) and IPC (serialization, data copy, context switches for PAL pipe communication) could easily dominate the actual GPU kernel time, especially for smaller vectors.
    *   As vector size increases, the GPU kernel time will increase. The relative overhead of masking/IPC might decrease, but the absolute time for these operations will also increase due to larger data volumes.
    *   **Vector Addition is Memory-Bound:** For smaller vector sizes, the actual GPU kernel execution time for vector addition is typically very short and dominated by memory transfer times (CPU to GPU, GPU to CPU).
    *   **Gramine-direct Overhead:** Expected to be low (e.g., 5-15% over Native). This overhead comes from the PAL syscall translation layer and the LibOS environment. Since vector addition involves data transfers that will pass through these layers, some impact is expected.
    *   **Gramine SGX Overhead:** Expected to be significantly higher, potentially several times the native execution time, especially for smaller vectors.
        *   **Data Masking (AES-GCM):** Encrypting two input vectors and decrypting one output vector (or vice-versa for the service) will add substantial CPU overhead. If `VECTOR_SIZE_ELEMENTS` is 2^20 floats (4MB), then 3 such operations mean 12MB of data processed by AES-GCM per call. This will be a major contributor.
        *   **IPC:** Transferring these (potentially large) masked data buffers between client and server enclaves via Gramine's IPC (which uses PAL pipes) involves data copies and context switches (ECALLs/OCALLs for PAL interaction), adding latency.
        *   **GPU Kernel vs. Overhead:** The actual `vectorAddKernel` execution on the GPU might be very fast (e.g., sub-millisecond to a few milliseconds). The masking and IPC overheads could easily be tens or hundreds of milliseconds, thus dominating the end-to-end time.
    *   **Impact of Vector Size:**
        *   As vector size increases, the GPU kernel time and data transfer times (PCIe) will increase.
        *   The AES-GCM processing time will also scale linearly with vector size.
        *   The IPC data copy time will scale linearly with vector size.
        *   The *relative* overhead of fixed IPC setup costs and context switches might decrease for very large vectors, but the data-proportional overheads (masking, copying) will remain significant.
    *   **SGX Time Breakdown (Table 3):**
        *   `Client: Data Prep & Masking` and `Shared Enc: IPC Recv & Unmask` (and their counterparts for results) would likely be the largest components.
        *   `Client: IPC Send + Wait` would also be significant, encompassing the round-trip latency.
        *   `Shared Enc: GPU Execution (CUDA)` might be a surprisingly small portion of the total SGX time unless the vectors are extremely large, making the GPU work itself substantial.

ONNX Model Inference (MobileNetV2)
----------------------------------

**Table 4: ONNX MobileNetV2 - End-to-End Execution Time (seconds)**

+----------------------+----------+-----------------+---------------------------+
| Workload             | Native   | Gramine-direct  | Gramine SGX (Shared Enc.) |
+======================+==========+=================+===========================+
| MobileNetV2 (1x3x224x224)| [time_n_onnx]| [time_gd_onnx]  | [time_sgx_onnx]           |
+----------------------+----------+-----------------+---------------------------+

**Table 5: ONNX MobileNetV2 - Calculated Overheads vs Native**

+----------------------+---------------------------+-----------------------------------+
| Workload             | Gramine-direct Overhead % | Gramine SGX (Shared Enc.) Ovhd. % |
+======================+===========================+===================================+
| MobileNetV2          | [ovhd_gd_onnx]%           | [ovhd_sgx_onnx]%                  |
+----------------------+---------------------------+-----------------------------------+

**Table 6: ONNX MobileNetV2 - SGX Mode Time Breakdown (ms - Hypothetical)**

+---------------------------------+------------+
| Component                       | Time (ms)  |
+=================================+============+
| Client: Input Prep & Masking    | [time_c_prep_mask_onnx] |
+---------------------------------+------------+
| Client: IPC Send + Wait         | [time_c_ipc_onnx]       |
+---------------------------------+------------+
| Client: Output Unmask & Process | [time_c_unmask_ver_onnx] |
+---------------------------------+------------+
| Shared Enc: IPC Recv & Unmask | [time_s_ipc_unmask_onnx] |
+---------------------------------+------------+
| Shared Enc: GPU Execution (ORT) | [time_s_gpu_onnx]       |
+---------------------------------+------------+
| Shared Enc: Output Mask & IPC Send| [time_s_mask_ipc_onnx]  |
+---------------------------------+------------+
| **Client End-to-End Total**     | [time_c_total_onnx]     |
+---------------------------------+------------+

**Hypothetical Discussion Points (ONNX MobileNetV2):**
    *   MobileNetV2 involves a mix of compute and memory operations. The GPU execution time via ONNX Runtime (ORT) will be more substantial than a simple vector add.
    *   Input tensor size (1x3x224x224 floats ~600KB) and output tensor size (1000 floats ~4KB) are fixed.
    *   **Gramine-direct overhead:** Should still be relatively low.
    *   **Gramine SGX overhead:** The data masking and IPC for the input tensor will be a noticeable fixed cost. The output tensor masking/IPC is much smaller.
    *   The relative overhead in SGX mode might be lower than for very quick vector additions because the `Shared Enc: GPU Execution (ORT)` time is expected to be larger. However, if the ORT execution itself is very fast (e.g., < 50ms), the masking/IPC overheads will still be a significant percentage.
    *   **Mixed CPU/GPU Workload:** ONNX Runtime (ORT) with the CUDA Execution Provider (EP) will perform some operations on the CPU (e.g., model loading, some pre/post-processing operators if not GPU-compatible) and offload compatible computations to the GPU. MobileNetV2 is relatively small.
    *   **Data Sizes:** Input (1x3x224x224 floats) is approx. 600KB. Output (1000 floats) is approx. 4KB.
    *   **Gramine-direct Overhead:** Expected to be low to moderate. ORT itself has a complex initialization phase; how Gramine handles file access for model loading and library loading might contribute.
    *   **Gramine SGX Overhead:**
        *   **Data Masking:** Masking the ~600KB input and ~4KB output. The input masking will be the more significant part here.
        *   **IPC:** Transferring these buffers.
        *   **ORT Initialization within SGX:** If the ONNX model is loaded and the session is initialized on each call (less likely for a persistent service), this would add significant overhead within the shared enclave. Assuming the service initializes ORT once.
        *   **GPU Execution:** The actual GPU inference time for MobileNetV2 is typically in the order of a few milliseconds on modern GPUs.
    *   **Relative Overhead:** The masking/IPC overhead for the ~600KB input will be noticeable. If the native GPU inference is very fast (e.g., 1-5ms), the relative SGX overhead could be high (e.g., 5x-20x or more). If the model were much larger and took longer on the GPU, the relative overhead would decrease.
    *   **SGX Time Breakdown (Table 6):**
        *   `Client: Input Prep & Masking` and `Shared Enc: IPC Recv & Unmask` for the input tensor are expected to be significant.
        *   `Shared Enc: GPU Execution (ORT)` would be the core ORT inference time on GPU.
        *   Masking/IPC for the smaller output tensor would be less impactful than for the input.

cuBLAS GEMM (SGEMM)
-------------------

**Table 7: cuBLAS SGEMM - End-to-End Execution Time (seconds)**

+----------------------+----------+-----------------+---------------------------+
| Workload (MxN, K)    | Native   | Gramine-direct  | Gramine SGX (Shared Enc.) |
+======================+==========+=================+===========================+
| 512x512, K=512       | [time_n_g1]| [time_gd_g1]    | [time_sgx_g1]             |
+----------------------+----------+-----------------+---------------------------+
| 1024x1024, K=1024    | [time_n_g2]| [time_gd_g2]    | [time_sgx_g2]             |
+----------------------+----------+-----------------+---------------------------+
| 2048x2048, K=2048    | [time_n_g3]| [time_gd_g3]    | [time_sgx_g3]             |
+----------------------+----------+-----------------+---------------------------+

**Table 8: cuBLAS SGEMM - Calculated Overheads vs Native**

+----------------------+---------------------------+-----------------------------------+
| Workload (MxN, K)    | Gramine-direct Overhead % | Gramine SGX (Shared Enc.) Ovhd. % |
+======================+===========================+===================================+
| 512x512, K=512       | [ovhd_gd_g1]%             | [ovhd_sgx_g1]%                    |
+----------------------+---------------------------+-----------------------------------+
| 1024x1024, K=1024    | [ovhd_gd_g2]%             | [ovhd_sgx_g2]%                    |
+----------------------+---------------------------+-----------------------------------+
| 2048x2048, K=2048    | [ovhd_gd_g3]%             | [ovhd_sgx_g3]%                    |
+----------------------+---------------------------+-----------------------------------+

**Table 9: cuBLAS SGEMM @ 1024x1024, K=1024 - SGX Mode Time Breakdown (ms - Hypothetical)**

+---------------------------------+------------+
| Component                       | Time (ms)  |
+=================================+============+
| Client: Matrix Prep & Masking   | [time_c_prep_mask_gemm] |
+---------------------------------+------------+
| Client: IPC Send + Wait         | [time_c_ipc_gemm]       |
+---------------------------------+------------+
| Client: Result Unmask & Verify  | [time_c_unmask_ver_gemm] |
+---------------------------------+------------+
| Shared Enc: IPC Recv & Unmask | [time_s_ipc_unmask_gemm] |
+---------------------------------+------------+
| Shared Enc: GPU Execution (cuBLAS)| [time_s_gpu_gemm]       |
+---------------------------------+------------+
| Shared Enc: Result Mask & IPC Send| [time_s_mask_ipc_gemm]  |
+---------------------------------+------------+
| **Client End-to-End Total**     | [time_c_total_gemm]     |
+---------------------------------+------------+

**Hypothetical Discussion Points (cuBLAS GEMM):**
    *   SGEMM is compute-intensive. GPU execution time will be substantial, especially for larger matrices.
    *   Data sizes for matrices can become very large (e.g., a 2048x2048 float matrix is 16MB). Transferring two such input matrices and one output matrix involves significant data movement and therefore significant AES-GCM processing time.
    *   **Gramine-direct overhead:** Expected to be minimal.
    *   **Gramine SGX overhead:**
        *   For smaller matrices (e.g., 512x512), the masking and IPC overhead might still be a large percentage of the total time if the cuBLAS call is very fast.
        *   For larger matrices (e.g., 2048x2048), the `Shared Enc: GPU Execution (cuBLAS)` time will likely dominate the SGX breakdown. The relative overhead of masking/IPC will decrease compared to the GPU time, but the absolute time for these security operations will be high due to the large data volumes.
        *   The efficiency of `libos_aes_gcm` implementation will be critical here. Hardware AES support (AES-NI) is essential for good performance.
    *   **Compute-Intensive:** SGEMM operations are highly compute-bound, especially for larger matrix dimensions. The GPU can be kept busy for a significant duration.
    *   **Data Sizes:** Matrix data can be very large. For M=N=K=512, each float matrix is 1MB. Two input matrices (A, B) and one output matrix (C) mean 3MB of data for AES-GCM processing and IPC per GEMM call if all are transferred. The example transfers A and B, gets C back.
    *   **Gramine-direct Overhead:** Expected to be minimal, as cuBLAS calls will pass through Gramine to the host driver with little interference for the computation itself.
    *   **Gramine SGX Overhead:**
        *   **Data Masking:** AES-GCM on potentially multiple megabytes of data per call will be a very significant overhead. For 512x512 matrices, this is 1MB for A, 1MB for B to be decrypted by the service, and 1MB for C to be encrypted. Total: 3MB per SGEMM operation.
        *   **IPC:** Transferring these large, masked buffers.
        *   **GPU Execution (cuBLAS):** For matrices like 512x512 or 1024x1024, the cuBLAS execution time will be substantial and will likely be the largest component in the SGX time breakdown, but the masking overhead will also be very large.
    *   **Impact of Matrix Size:**
        *   GPU time increases roughly with O(N^3).
        *   Masking and IPC time increase with O(N^2) (data size).
        *   Therefore, as matrix dimensions increase, the *relative* overhead of masking/IPC should decrease compared to the GPU computation time. However, the *absolute* time for masking/IPC will still be high and may become a bottleneck for system throughput if many such operations are done.
    *   **SGX Time Breakdown (Table 9 for 1024^2 matrices):**
        *   `Shared Enc: GPU Execution (cuBLAS)` is expected to be the largest single component.
        *   `Client: Matrix Prep & Masking`, `Shared Enc: IPC Recv & Unmask`, `Shared Enc: Result Mask & IPC Send`, and `Client: Result Unmask & Verify` will all be significant due to the 4MB per matrix (for 1024^2).
        *   The efficiency of AES-GCM (leveraging AES-NI) is paramount here.

4. Analysis and Bottleneck Identification (Hypothetical)
========================================================

Based on the hypothetical discussions above:

*   **Impact of Data Size on Overhead:**
    *   The overhead introduced by data masking (AES-GCM) and IPC is directly proportional to the size of the data being transferred. This is evident in all three benchmarks.
    *   For applications with large input/output data (e.g., large GEMM matrices, high-resolution images for ONNX models not discussed but applicable), AES-GCM processing and data copying for IPC will be major performance factors.
    *   The performance of `libos_aes_gcm.c` (and the underlying mbedTLS) is critical. Hardware AES acceleration (AES-NI) on the CPU is essential to keep these costs manageable.

*   **Impact of GPU Computation Intensity vs. Communication/Security Overheads:**
    *   **Low GPU Intensity (e.g., Small Vector Add):** Fixed costs of IPC setup, context switching (ECALLs/OCALLs), and data-proportional costs of masking/copying can easily dominate the very short GPU kernel time. This results in very high relative overhead for the SGX shared enclave model.
    *   **Moderate GPU Intensity (e.g., MobileNetV2 ONNX):** GPU execution time is more significant. Masking the input tensor (~600KB) and IPC will still be major contributors. The relative overhead will be lower than for small vector additions but still substantial.
    *   **High GPU Intensity (e.g., Large SGEMM):** GPU execution time becomes the dominant factor in the end-to-end latency. While masking and IPC for large matrices (several MBs) take considerable absolute time, their percentage contribution to the total end-to-end time decreases. The system approaches being limited by either GPU compute power or the data masking/transfer throughput.

*   **Primary Bottlenecks in SGX Shared Enclave Mode:**
    1.  **AES-GCM Processing:** For data-heavy workloads, the CPU time spent encrypting and decrypting data buffers is a primary bottleneck.
    2.  **IPC Data Transfer & Serialization:** Copying data between client and server enclave memory, even via efficient local pipes, incurs overhead. While the example uses direct struct copies, more complex serialization/deserialization would add to this.
    3.  **SGX Context Switching (ECALLs/OCALLs):** Each IPC message involves multiple transitions between enclaves and the untrusted runtime (for PAL calls). CUDA driver interactions from the shared enclave also involve ECALLs/OCALLs. While Gramine optimizes these, their frequency adds up. For instance, a single client request might involve: Client ECALL (ipc_send) -> PAL OCALL (pipe write) -> Host -> PAL OCALL (pipe read in server) -> Server ECALL (ipc_receive) -> Server ECALL (cuda calls) -> ... and similar path for response.
    4.  **Memory Allocation/Management:** Frequent allocation/deallocation of large buffers for plaintext/ciphertext within enclaves can add overhead, though the examples try to manage this. Using memory pools could be an optimization if this becomes an issue.
    5.  **PCIe Data Transfer:** While not directly an SGX overhead, the time to transfer data to/from GPU memory over PCIe is part of the "GPU Execution" component measured by `cudaEvent_t` and can be substantial for large data, affecting overall efficiency.

5. Conclusions and Recommendations (Hypothetical)
=================================================

*   **Performance Characteristics Summary:**
    *   **Gramine-direct:** Generally introduces low overhead (e.g., 5-20%) over native execution for GPU workloads. The primary sources of overhead are PAL syscall translations for file I/O (if any), device interactions, and the general LibOS environment.
    *   **Gramine SGX (Shared Enclave):** Introduces significant performance overhead compared to native or Gramine-direct. This overhead is primarily due to:
        *   AES-GCM data masking (encryption/decryption) for all data crossing the client-shared enclave boundary.
        *   Inter-enclave IPC for request/response messages, including data serialization and copies.
        *   Increased number of ECALLs/OCALLs due to the disaggregated architecture.
    *   The relative overhead in SGX mode is highest for tasks with short GPU execution times and/or small data transfers where the fixed and data-proportional security/communication costs dominate.
    *   For compute-intensive GPU tasks with large data, the GPU computation time becomes a larger portion of the total, making the relative overhead of the security mechanisms more acceptable, though absolute overheads for masking/IPC can still be high.

*   **Recommendations for Use:**
    *   The shared enclave GPU architecture is most suitable when strong isolation between multiple clients and a centralized, GPU-accessing service is required, and the data processed by the GPU warrants protection during transit to/from the shared enclave and while potentially in GPU memory (from untrusted OS/hypervisor).
    *   It is best applied to workloads where the GPU computation offloaded is substantial enough to amortize the overheads of data masking and IPC. Very frequent, small offloads will likely see poor performance.
    *   Consider batching multiple small requests from a client into a single larger IPC transaction to reduce per-request overhead.
    *   Optimize data structures for efficient masking and IPC. Only transfer necessary data.
    *   If data sensitivity allows, using untrusted shared memory (``fs.type = "untrusted_shm"``) for very large, non-sensitive data exchange could be an alternative to IPC, but this bypasses the data masking provided by the example's IPC mechanism.

*   **Potential Future Optimization Areas:**
    *   **AES-GCM Performance:** Investigate more performant AES-GCM implementations or alternative authenticated encryption schemes if they offer better throughput with similar security guarantees. Ensure full utilization of hardware AES-NI. Consider if mbedTLS version or build options impact this.
    *   **IPC Optimization:** Explore options to reduce data copies in Gramine's IPC mechanism. Look into potential for zero-copy mechanisms if feasible within SGX constraints (likely difficult across enclaves without shared memory, which then needs its own protection).
    *   **Asynchronous Operations & Batching:** Design client and server applications to use asynchronous IPC and GPU operations where possible to overlap computation with communication and masking. Batching requests can significantly reduce per-request overheads.
    *   **Selective Masking:** If parts of the data being transferred are non-sensitive, the protocol could be enhanced to allow selective masking, reducing cryptographic load. This requires careful design to ensure no sensitive data leaks.
    *   **Reduced Context Switching:** While fundamental, any improvements in Gramine's ECALL/OCALL efficiency or reducing the number of transitions needed for common operations (e.g., optimized CUDA driver interaction paths) would benefit this architecture.

6. Raw Data (Placeholder)
=========================

Actual timing data (from ``/usr/bin/time -v`` outputs like ``*.time`` files) and internal application logs (containing ``cudaEvent_t`` timings, etc.) would be collected from running the ``CI-Examples/run_benchmarks.sh`` script. These would be stored in the ``CI-Examples/benchmark_results/`` directory.

For a full report, these raw numbers would be processed (averaged, standard deviations calculated) and used to populate the tables in Section 3. Key snippets or summaries of ``/usr/bin/time -v`` output (e.g., User/System CPU time, Max RSS) would also be presented here or in an appendix to support the analysis.

*(End of gpu_shared_enclave_analysis.rst)*
