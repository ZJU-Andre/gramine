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
The purpose of this document is to analyze the performance characteristics and overheads associated with the shared enclave GPU architecture in Gramine. This architecture is designed to allow multiple client enclaves to securely utilize a GPU managed by a central shared service enclave. A key feature explored is selective data masking, allowing for performance trade-offs based on data sensitivity.

Architecture Overview
---------------------
The architecture under test involves:
*   **Client Enclaves:** Applications running in Gramine that require GPU-accelerated computations.
*   **Shared Enclave:** A dedicated Gramine enclave that has direct access to the host GPU devices (e.g., ``/dev/nvidia*``). It receives computation requests from client enclaves.
*   **Inter-Process Communication (IPC):** Client enclaves communicate with the shared enclave using Gramine's IPC mechanism (based on PAL pipes).
*   **Selective Data Masking (AES-GCM / None):** Data transferred between client enclaves and the shared enclave for GPU processing can be either:
    *   Protected using AES-GCM encryption and authentication (`MASKING_AES_GCM`).
    *   Transferred as plaintext (`MASKING_NONE`) if deemed non-sensitive for direct GPU interaction.
    The shared enclave is responsible for unmasking data (if encrypted) before GPU operations and re-masking results (if required by the chosen mode).

This analysis aims to quantify the performance impact of these components, particularly the IPC and data masking overheads, relative to native execution, a non-SGX Gramine (direct mode) execution, and between the different masking modes within SGX.

2. Benchmarking Methodology
===========================

Test Applications
-----------------
(Unchanged from previous version - Vector Addition, ONNX MobileNetV2, cuBLAS GEMM)

Execution Modes
---------------
Each benchmark application is run in the following modes:

1.  **Native Linux:** Baseline.
2.  **Gramine-direct:** Isolates Gramine LibOS/PAL overhead without SGX.
3.  **Gramine SGX (AES-GCM Masking):** Shared enclave model with full AES-GCM encryption/decryption of GPU data payloads.
4.  **Gramine SGX (No GPU Masking / Plaintext):** Shared enclave model where GPU data payloads are transferred as plaintext between client and shared enclave (via IPC). Other IPC control data might still be implicitly protected by the IPC channel itself, but the large GPU buffers are not explicitly encrypted by the application layer.

Metrics Collected
-----------------
(Unchanged from previous version - End-to-End Time, Internal GPU Time, SGX Mode Time Breakdown, System-Level Stats)
The SGX Mode Time Breakdown will be particularly important to compare between "AES-GCM Masking" and "No GPU Masking" modes.

Hardware/Software Environment
-----------------------------
(Unchanged - User to fill details)

Benchmark Execution
-------------------
(Unchanged - Multiple runs, averaging, warm-up considerations)

3. Performance Results (Hypothetical Discussion & Templates)
============================================================

This section presents templates for reporting performance results and discusses hypothetical outcomes and expectations. **Actual benchmark data needs to be collected by running the scripts.**

Vector Addition
---------------

**Table 1: Vector Addition - End-to-End Execution Time (seconds)**

+----------------------+----------+-----------------+---------------------------+---------------------------------+
| Workload (Elements)  | Native   | Gramine-direct  | Gramine SGX (AES-GCM)     | Gramine SGX (No GPU Masking)  |
+======================+==========+=================+===========================+=================================+
| 2^20 (approx 1M)     | [time_n1]| [time_gd1]      | [time_sgx_aes1]           | [time_sgx_none1]                |
+----------------------+----------+-----------------+---------------------------+---------------------------------+
| 2^22 (approx 4M)     | [time_n2]| [time_gd2]      | [time_sgx_aes2]           | [time_sgx_none2]                |
+----------------------+----------+-----------------+---------------------------+---------------------------------+
| 2^24 (approx 16M)    | [time_n3]| [time_gd3]      | [time_sgx_aes3]           | [time_sgx_none3]                |
+----------------------+----------+-----------------+---------------------------+---------------------------------+

**Table 2: Vector Addition - Calculated Overheads vs Native**

+----------------------+---------------------------+-----------------------------------+----------------------------------------+
| Workload (Elements)  | Gramine-direct Overhead % | Gramine SGX (AES-GCM) Ovhd. %     | Gramine SGX (No GPU Masking) Ovhd. % |
+======================+===========================+===================================+========================================+
| 2^20                 | [ovhd_gd1]%               | [ovhd_sgx_aes1]%                  | [ovhd_sgx_none1]%                      |
+----------------------+---------------------------+-----------------------------------+----------------------------------------+
| 2^22                 | [ovhd_gd2]%               | [ovhd_sgx_aes2]%                  | [ovhd_sgx_none2]%                      |
+----------------------+---------------------------+-----------------------------------+----------------------------------------+
| 2^24                 | [ovhd_gd3]%               | [ovhd_sgx_aes3]%                  | [ovhd_sgx_none3]%                      |
+----------------------+---------------------------+-----------------------------------+----------------------------------------+

**Table 3.1: Vector Addition @ 2^22 Elements - SGX (AES-GCM Masking) Time Breakdown (ms - Hypothetical)**

+---------------------------------+------------+
| Component                       | Time (ms)  |
+=================================+============+
| Client: Data Prep & Masking     | [time_c_prep_mask_va] |
+---------------------------------+------------+
| Client: IPC Send + Wait         | [time_c_ipc_va_aes]   |
+---------------------------------+------------+
| Client: Data Unmask & Verify  | [time_c_unmask_ver_va] |
+---------------------------------+------------+
| Shared Enc: IPC Recv & Unmask | [time_s_ipc_unmask_va] |
+---------------------------------+------------+
| Shared Enc: GPU Execution (CUDA)| [time_s_gpu_va]       |
+---------------------------------+------------+
| Shared Enc: Data Mask & IPC Send| [time_s_mask_ipc_va]  |
+---------------------------------+------------+
| **Client End-to-End Total**     | [time_c_total_va_aes] |
+---------------------------------+------------+

**Table 3.2: Vector Addition @ 2^22 Elements - SGX (No GPU Masking) Time Breakdown (ms - Hypothetical)**

+---------------------------------+------------+
| Component                       | Time (ms)  |
+=================================+============+
| Client: Data Prep (No Masking)  | [time_c_prep_nomask_va] | <!-- Primarily memcpy -->
+---------------------------------+------------+
| Client: IPC Send + Wait         | [time_c_ipc_va_none]  |
+---------------------------------+------------+
| Client: Data Verify (No Unmask) | [time_c_ver_nounmask_va] |
+---------------------------------+------------+
| Shared Enc: IPC Recv (No Unmask)| [time_s_ipc_nounmask_va] |
+---------------------------------+------------+
| Shared Enc: GPU Execution (CUDA)| [time_s_gpu_va]       | <!-- Same as AES-GCM case -->
+---------------------------------+------------+
| Shared Enc: Data Prep (No Mask) | [time_s_prep_nomask_va] | <!-- Primarily memcpy for response -->
+---------------------------------+------------+
| **Client End-to-End Total**     | [time_c_total_va_none]|
+---------------------------------+------------+

**Hypothetical Discussion Points (Vector Addition):**
    *   (Discussion on Native and Gramine-direct from previous version remains relevant)
    *   **Gramine SGX (AES-GCM) Overhead:** As discussed before, this is expected to be high due to masking and IPC for large vector data relative to fast kernel execution.
    *   **Gramine SGX (No GPU Masking) Overhead:** This mode eliminates the AES-GCM processing time. The overhead compared to Gramine-direct will primarily be due to IPC data copying and SGX transitions. Expected to be significantly faster than AES-GCM mode for this benchmark.
    *   **Comparison:** The difference between `[time_sgx_aesN]` and `[time_sgx_noneN]` will directly show the cost of AES-GCM for 3*N*elements*sizeof(float) data.
    *   **Breakdown Analysis:** For "No GPU Masking", the masking/unmasking components in Table 3.2 will be near zero or represent simple memory copy times. The IPC time might also be slightly lower if not sending IVs/tags, but the bulk data copy for payloads remains.

ONNX Model Inference (MobileNetV2)
----------------------------------

**Table 4: ONNX MobileNetV2 - End-to-End Execution Time (seconds)**

+----------------------+----------+-----------------+---------------------------+---------------------------------+
| Workload             | Native   | Gramine-direct  | Gramine SGX (AES-GCM)     | Gramine SGX (No GPU Masking)  |
+======================+==========+=================+===========================+=================================+
| MobileNetV2          | [time_n_onnx]| [time_gd_onnx]  | [time_sgx_aes_onnx]       | [time_sgx_none_onnx]            |
+----------------------+----------+-----------------+---------------------------+---------------------------------+

**Table 5: ONNX MobileNetV2 - Calculated Overheads vs Native**

+----------------------+---------------------------+-----------------------------------+----------------------------------------+
| Workload             | Gramine-direct Overhead % | Gramine SGX (AES-GCM) Ovhd. %     | Gramine SGX (No GPU Masking) Ovhd. % |
+======================+===========================+===================================+========================================+
| MobileNetV2          | [ovhd_gd_onnx]%           | [ovhd_sgx_aes_onnx]%              | [ovhd_sgx_none_onnx]%                  |
+----------------------+---------------------------+-----------------------------------+----------------------------------------+

**Table 6.1: ONNX MobileNetV2 - SGX (AES-GCM Masking) Time Breakdown (ms - Hypothetical)**

+---------------------------------+------------+
| Component                       | Time (ms)  |
+=================================+============+
| Client: Input Prep & Masking    | [time_c_prep_mask_onnx] |
+---------------------------------+------------+
| Client: IPC Send + Wait         | [time_c_ipc_onnx_aes]   |
+---------------------------------+------------+
| Client: Output Unmask & Process | [time_c_unmask_ver_onnx] |
+---------------------------------+------------+
| Shared Enc: IPC Recv & Unmask | [time_s_ipc_unmask_onnx] |
+---------------------------------+------------+
| Shared Enc: GPU Execution (ORT) | [time_s_gpu_onnx]       |
+---------------------------------+------------+
| Shared Enc: Output Mask & IPC Send| [time_s_mask_ipc_onnx]  |
+---------------------------------+------------+
| **Client End-to-End Total**     | [time_c_total_onnx_aes] |
+---------------------------------+------------+

**Table 6.2: ONNX MobileNetV2 - SGX (No GPU Masking) Time Breakdown (ms - Hypothetical)**

+---------------------------------+------------+
| Component                       | Time (ms)  |
+=================================+============+
| Client: Input Prep (No Masking) | [time_c_prep_nomask_onnx] |
+---------------------------------+------------+
| Client: IPC Send + Wait         | [time_c_ipc_onnx_none]  |
+---------------------------------+------------+
| Client: Output Process(No Unmask)| [time_c_ver_nounmask_onnx] |
+---------------------------------+------------+
| Shared Enc: IPC Recv (No Unmask)| [time_s_ipc_nounmask_onnx] |
+---------------------------------+------------+
| Shared Enc: GPU Execution (ORT) | [time_s_gpu_onnx]       |
+---------------------------------+------------+
| Shared Enc: Output Prep (No Mask)| [time_s_prep_nomask_onnx] |
+---------------------------------+------------+
| **Client End-to-End Total**     | [time_c_total_onnx_none]|
+---------------------------------+------------+

**Hypothetical Discussion Points (ONNX MobileNetV2):**
    *   (Discussion on Native and Gramine-direct from previous version remains relevant)
    *   **Gramine SGX (AES-GCM) Overhead:** Masking the ~600KB input tensor will be the primary crypto overhead. Output is small.
    *   **Gramine SGX (No GPU Masking) Overhead:** Eliminates AES-GCM for the ~600KB input and ~4KB output. The performance gain should be noticeable and primarily reflect the cost of encrypting/decrypting the input tensor.
    *   **Comparison:** The difference between SGX modes will quantify the AES-GCM cost for ~604KB of data. Given that GPU execution for MobileNetV2 is fast, this saving can be a significant portion of the SGX overhead.
    *   **Breakdown Analysis:** Table 6.2 will show near-zero times for masking/unmasking components. The core GPU time remains the same.

cuBLAS GEMM (SGEMM)
-------------------

**Table 7: cuBLAS SGEMM - End-to-End Execution Time (seconds)**

+----------------------+----------+-----------------+---------------------------+---------------------------------+
| Workload (MxN, K)    | Native   | Gramine-direct  | Gramine SGX (AES-GCM)     | Gramine SGX (No GPU Masking)  |
+======================+==========+=================+===========================+=================================+
| 512x512, K=512       | [time_n_g1]| [time_gd_g1]    | [time_sgx_aes_g1]         | [time_sgx_none_g1]              |
+----------------------+----------+-----------------+---------------------------+---------------------------------+
| 1024x1024, K=1024    | [time_n_g2]| [time_gd_g2]    | [time_sgx_aes_g2]         | [time_sgx_none_g2]              |
+----------------------+----------+-----------------+---------------------------+---------------------------------+
| 2048x2048, K=2048    | [time_n_g3]| [time_gd_g3]    | [time_sgx_aes_g3]         | [time_sgx_none_g3]              |
+----------------------+----------+-----------------+---------------------------+---------------------------------+

**Table 8: cuBLAS SGEMM - Calculated Overheads vs Native**

+----------------------+---------------------------+-----------------------------------+----------------------------------------+
| Workload (MxN, K)    | Gramine-direct Overhead % | Gramine SGX (AES-GCM) Ovhd. %     | Gramine SGX (No GPU Masking) Ovhd. % |
+======================+===========================+===================================+========================================+
| 512x512, K=512       | [ovhd_gd_g1]%             | [ovhd_sgx_aes_g1]%                | [ovhd_sgx_none_g1]%                    |
+----------------------+---------------------------+-----------------------------------+----------------------------------------+
| 1024x1024, K=1024    | [ovhd_gd_g2]%             | [ovhd_sgx_aes_g2]%                | [ovhd_sgx_none_g2]%                    |
+----------------------+---------------------------+-----------------------------------+----------------------------------------+
| 2048x2048, K=2048    | [ovhd_gd_g3]%             | [ovhd_sgx_aes_g3]%                | [ovhd_sgx_none_g3]%                    |
+----------------------+---------------------------+-----------------------------------+----------------------------------------+

**Table 9.1: cuBLAS SGEMM @ 1024x1024, K=1024 - SGX (AES-GCM Masking) Time Breakdown (ms - Hypothetical)**

+---------------------------------+------------+
| Component                       | Time (ms)  |
+=================================+============+
| Client: Matrix Prep & Masking   | [time_c_prep_mask_gemm] |
+---------------------------------+------------+
| Client: IPC Send + Wait         | [time_c_ipc_gemm_aes]   |
+---------------------------------+------------+
| Client: Result Unmask & Verify  | [time_c_unmask_ver_gemm] |
+---------------------------------+------------+
| Shared Enc: IPC Recv & Unmask | [time_s_ipc_unmask_gemm] |
+---------------------------------+------------+
| Shared Enc: GPU Execution (cuBLAS)| [time_s_gpu_gemm]       |
+---------------------------------+------------+
| Shared Enc: Result Mask & IPC Send| [time_s_mask_ipc_gemm]  |
+---------------------------------+------------+
| **Client End-to-End Total**     | [time_c_total_gemm_aes] |
+---------------------------------+------------+

**Table 9.2: cuBLAS SGEMM @ 1024x1024, K=1024 - SGX (No GPU Masking) Time Breakdown (ms - Hypothetical)**

+---------------------------------+------------+
| Component                       | Time (ms)  |
+=================================+============+
| Client: Matrix Prep (No Masking)| [time_c_prep_nomask_gemm] |
+---------------------------------+------------+
| Client: IPC Send + Wait         | [time_c_ipc_gemm_none]  |
+---------------------------------+------------+
| Client: Result Verify (No Unmask)| [time_c_ver_nounmask_gemm] |
+---------------------------------+------------+
| Shared Enc: IPC Recv (No Unmask)| [time_s_ipc_nounmask_gemm] |
+---------------------------------+------------+
| Shared Enc: GPU Execution (cuBLAS)| [time_s_gpu_gemm]       |
+---------------------------------+------------+
| Shared Enc: Result Prep (No Mask)| [time_s_prep_nomask_gemm] |
+---------------------------------+------------+
| **Client End-to-End Total**     | [time_c_total_gemm_none]|
+---------------------------------+------------+

**Hypothetical Discussion Points (cuBLAS GEMM):**
    *   (Discussion on Native and Gramine-direct from previous version remains relevant)
    *   **Gramine SGX (AES-GCM) Overhead:** For large matrices, AES-GCM processing on multiple megabytes (e.g., 2MB for inputs, 1MB for output for 512^2; 8MB inputs, 4MB output for 1024^2) will be very substantial.
    *   **Gramine SGX (No GPU Masking) Overhead:** This will show a significant improvement over AES-GCM mode. The remaining overhead vs. Gramine-direct will be due to IPC of large plaintext matrices and SGX transitions.
    *   **Comparison:** The time difference between `[time_sgx_aes_gN]` and `[time_sgx_none_gN]` will highlight the direct cost of cryptographic protection for large data arrays.
    *   **Breakdown Analysis:** In "No GPU Masking" mode (Table 9.2), the masking/unmasking times become negligible. The IPC time might still be high due to large data copies. The GPU execution time remains the core compute part. For very large matrices, GPU time will still likely dominate even in "No GPU Masking" SGX mode.

4. Analysis and Bottleneck Identification (Hypothetical)
========================================================

(The existing discussion on "Impact of Data Size on Overhead", "Impact of GPU Computation Intensity vs. Communication/Security Overheads", and "Primary Bottlenecks in SGX Shared Enclave Mode" remains largely relevant.)

**NEW SUBSECTION:**

Impact of Selective Data Masking (AES-GCM vs. No Masking)
---------------------------------------------------------
The introduction of `gpu_data_masking_level_t` allows for a direct comparison of performance when AES-GCM data masking is enabled versus when data is transferred as plaintext for GPU operations within the SGX shared enclave model.

*   **Performance Gains from `MASKING_NONE`:**
    *   The most significant performance gain from using `MASKING_NONE` is the elimination of CPU-bound AES-GCM encryption and decryption operations on the data payloads (input and output) for each GPU task. This gain is directly proportional to the size of the data being processed. For benchmarks like GEMM with large matrices or ONNX with large input tensors, the time saved from crypto operations can be substantial (potentially reducing this component of overhead to near zero).
    *   Vector Addition, especially with smaller vectors, might see a very large *relative* improvement when switching to `MASKING_NONE`, as crypto overhead likely dominated its AES-GCM SGX runtime. For ONNX and GEMM, while the absolute time saved by omitting crypto will be larger due to larger data sizes, the *relative* percentage improvement in end-to-end time might be more moderate if the GPU computation itself is already a large portion of the total time.
*   **Remaining Overheads in `MASKING_NONE` SGX Mode:**
    *   **IPC Data Transfer:** Data (now plaintext) still needs to be copied between the client and shared enclaves via the IPC mechanism. This involves memory copies and context switches (ECALLs/OCALLs for PAL pipe operations). For large data, this remains a bottleneck.
    *   **SGX Transitions:** ECALLs/OCALLs for IPC and for the shared enclave to interact with the CUDA driver (e.g., `cudaMalloc`, `cudaMemcpy`, kernel launch, `cublasSgemm`, `OrtRun`) are still incurred. These contribute to latency.
    *   **LibOS/PAL Overheads:** The general overhead of running within the Gramine LibOS and PAL layers (syscall translation, internal bookkeeping) still applies.
*   **Security Implications of `MASKING_NONE`:**
    *   It is crucial to reiterate that `MASKING_NONE` means that the GPU data payloads are transferred between enclaves as plaintext and will reside in GPU memory as plaintext. This data is vulnerable to observation by a compromised host OS/hypervisor while on the PCIe bus or in GPU device memory.
    *   This mode should **only** be used if the specific data being processed by the GPU is deemed non-sensitive, or if other external factors mitigate the risk (e.g., a physically secured system where host threats are not a concern for this specific data, though this typically negates the primary motivation for using SGX).
    *   The decision to use `MASKING_NONE` must be a careful trade-off between performance needs and the security requirements of the data and application. It effectively reduces the TCB for that specific data from SGX protection down to the security level of the GPU hardware and driver stack during GPU processing.

5. Conclusions and Recommendations (Hypothetical)
=================================================

*   **Performance Characteristics Summary:**
    *   (Gramine-direct conclusions remain the same)
    *   Gramine SGX (Shared Enclave) with `MASKING_AES_GCM` provides the highest level of data protection for GPU payloads exchanged with the shared enclave but incurs significant performance overhead due to cryptography and IPC.
    *   Gramine SGX (Shared Enclave) with `MASKING_NONE` offers a substantial performance improvement over the AES-GCM mode by eliminating cryptographic overhead. The remaining overhead compared to Gramine-direct is primarily due to IPC data copies and SGX transition costs.
*   **Recommendations for Use:**
    *   (Recommendations for AES-GCM mode remain the same)
    *   The `MASKING_NONE` option within the SGX shared enclave model should be used cautiously. It is appropriate for scenarios where:
        *   The specific data being offloaded to the GPU is determined to be non-sensitive or of sufficiently low value that its exposure outside the CPU TEE boundary (on PCIe bus, in GPU RAM) is an acceptable risk.
        *   Performance is critical, and the overhead of AES-GCM is prohibitive for the application's requirements.
    *   **A thorough risk assessment is essential before opting for `MASKING_NONE` for any production data.** The security guarantees for data processed via this path are significantly different from data processed under `MASKING_AES_GCM`.
    *   Even with `MASKING_NONE`, the shared enclave architecture still provides isolation for the service logic itself and can protect other sensitive assets within the shared enclave (like the ONNX model parameters if they are not part of the "masked" data, or cryptographic keys used for other purposes).
*   **Potential Future Optimization Areas:**
    *   (Existing points remain relevant)
    *   **Hybrid Masking:** For complex data structures, explore options where only specific sensitive fields are masked, while bulk non-sensitive data is transferred as plaintext. This would require more complex data handling and (de)serialization.

6. Raw Data (Placeholder)
=========================
(Unchanged)

*(End of gpu_shared_enclave_analysis.rst)*
