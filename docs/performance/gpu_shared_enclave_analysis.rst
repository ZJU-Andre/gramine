.. SPDX-License-Identifier: LGPL-3.0-or-later
.. Copyright (C) 2023-2024 Intel Corporation

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
The purpose of this document is to analyze the performance characteristics and overheads associated with the shared enclave GPU architecture in Gramine. This architecture is designed to allow multiple client enclaves to securely utilize a GPU managed by a central shared service enclave. Key features explored include selective data masking (AES-GCM for high sensitivity, plaintext with optional Direct Memory Access (DMA) for low sensitivity) and their performance implications.

Architecture Overview
---------------------
The architecture under test involves:
*   **Client Enclaves:** Applications running in Gramine that require GPU-accelerated computations.
*   **Shared Enclave:** A dedicated Gramine enclave that has direct access to the host GPU devices (e.g., ``/dev/nvidia*``). It receives computation requests from client enclaves.
*   **Inter-Process Communication (IPC):** Client enclaves communicate with the shared enclave using Gramine's IPC mechanism (based on PAL pipes).
*   **Selective Data Handling & Masking:** Data transferred between client enclaves and the shared enclave for GPU processing can be:
    *   **Protected using AES-GCM (`MASKING_AES_GCM`):** Input and output data payloads are encrypted and authenticated. The shared enclave unmasks data before GPU operations and re-masks results. This provides the highest data confidentiality and integrity during transfer and within the shared enclave's CPU memory.
    *   **Transferred as Plaintext with DMA (`MASKING_NONE` with client-provided device pointers):** For low-sensitivity data, clients can allocate pinned host memory, obtain device pointers to this memory, and send these pointers to the shared enclave. The shared enclave uses these device pointers directly (e.g., for VectorAdd, GEMM inputs) or via a Device-to-Device (DtoD) copy into an enclave-managed device buffer (e.g., for ONNX inputs) for GPU operations. This path aims to minimize data copies and CPU overhead.
    *   **Transferred as Plaintext in IPC Payload (`MASKING_NONE` without client-provided device pointers):** If device pointers are not provided, plaintext data is copied into the IPC message and then from the shared enclave's CPU memory to the GPU. This is less performant than DMA but still avoids cryptographic overhead.

This analysis aims to quantify the performance impact of these components, particularly IPC, data masking/copying overheads, and the benefits of the DMA path, relative to native execution and a non-SGX Gramine (direct mode) execution.

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
3.  **Gramine SGX (AES-GCM Masking):** Shared enclave model with full AES-GCM encryption/decryption of GPU data payloads. Data is copied into and out of the shared enclave.
4.  **Gramine SGX (`MASKING_NONE` with DMA):** Shared enclave model where client-provided device pointers for input data are used by the service. Data is considered plaintext from the client's perspective for GPU processing. This is the primary "No GPU Masking" mode discussed for performance benefits.
5.  **(Optional) Gramine SGX (`MASKING_NONE` via IPC Copy):** Shared enclave model where plaintext GPU data payloads are copied via IPC between client and shared enclave. This mode is implicitly used if `MASKING_NONE` is selected but the client does not provide device pointers. It serves as a comparison point to isolate DMA benefits from mere crypto elimination.

Metrics Collected
-----------------
(Unchanged from previous version - End-to-End Time, Internal GPU Time, SGX Mode Time Breakdown, System-Level Stats)
The SGX Mode Time Breakdown will be particularly important to compare between "AES-GCM Masking", "No GPU Masking (DMA)", and potentially "No GPU Masking (IPC Copy)" modes.

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

+----------------------+----------+-----------------+---------------------------+---------------------------------------+
| Workload (Elements)  | Native   | Gramine-direct  | Gramine SGX (AES-GCM)     | Gramine SGX (`MASKING_NONE` with DMA) |
+======================+==========+=================+===========================+=======================================+
| 2^20 (approx 1M)     | [time_n1]| [time_gd1]      | [time_sgx_aes1]           | [time_sgx_dma1]                       |
+----------------------+----------+-----------------+---------------------------+---------------------------------------+
| 2^22 (approx 4M)     | [time_n2]| [time_gd2]      | [time_sgx_aes2]           | [time_sgx_dma2]                       |
+----------------------+----------+-----------------+---------------------------+---------------------------------------+
| 2^24 (approx 16M)    | [time_n3]| [time_gd3]      | [time_sgx_aes3]           | [time_sgx_dma3]                       |
+----------------------+----------+-----------------+---------------------------+---------------------------------------+

**Table 2: Vector Addition - Calculated Overheads vs Native**

+----------------------+---------------------------+-----------------------------------+-----------------------------------------------+
| Workload (Elements)  | Gramine-direct Overhead % | Gramine SGX (AES-GCM) Ovhd. %     | Gramine SGX (`MASKING_NONE` with DMA) Ovhd. % |
+======================+===========================+===================================+===============================================+
| 2^20                 | [ovhd_gd1]%               | [ovhd_sgx_aes1]%                  | [ovhd_sgx_dma1]%                              |
+----------------------+---------------------------+-----------------------------------+-----------------------------------------------+
| 2^22                 | [ovhd_gd2]%               | [ovhd_sgx_aes2]%                  | [ovhd_sgx_dma2]%                              |
+----------------------+---------------------------+-----------------------------------+-----------------------------------------------+
| 2^24                 | [ovhd_gd3]%               | [ovhd_sgx_aes3]%                  | [ovhd_sgx_dma3]%                              |
+----------------------+---------------------------+-----------------------------------+-----------------------------------------------+

**Table 3.1: Vector Addition @ 2^22 Elements - SGX (AES-GCM Masking) Time Breakdown (ms - Hypothetical)**
*(Existing table structure is fine, ensure components reflect client-side masking and shared enclave unmasking + masking of results)*

**Table 3.2: Vector Addition @ 2^22 Elements - SGX (`MASKING_NONE` with DMA) Time Breakdown (ms - Hypothetical)**

+---------------------------------+------------+
| Component                       | Time (ms)  |
+=================================+============+
| Client: Pinned Mem Alloc & Ptr  | [time_c_pinalloc_va] |
+---------------------------------+------------+
| Client: IPC Send (Pointers) + Wait| [time_c_ipc_va_dma]   |
+---------------------------------+------------+
| Client: Result Verify (No Unmask)| [time_c_ver_nounmask_va] |
+---------------------------------+------------+
| Shared Enc: IPC Recv (Pointers) | [time_s_ipc_recv_ptrs_va]|
+---------------------------------+------------+
| Shared Enc: GPU Exec (Direct DMA)| [time_s_gpu_va_dma]   | <!-- Kernel uses client pointers -->
+---------------------------------+------------+
| Shared Enc: Result Prep (No Mask)| [time_s_prep_res_va]  | <!-- Memcpy result to response payload -->
+---------------------------------+------------+
| **Client End-to-End Total**     | [time_c_total_va_dma] |
+---------------------------------+------------+

**Hypothetical Discussion Points (Vector Addition):**
    *   (Discussion on Native and Gramine-direct from previous version remains relevant)
    *   **Gramine SGX (AES-GCM) Overhead:** As discussed before, this is expected to be high due to masking and IPC for large vector data relative to fast kernel execution.
    *   **Gramine SGX (`MASKING_NONE` with DMA) Overhead:** This mode eliminates AES-GCM processing and also avoids copying the large input data buffers (B and C) into the shared enclave's memory and then again to the GPU. The overhead compared to Gramine-direct will primarily be due to SGX transitions for IPC and CUDA API calls from the shared enclave, and the initial pinned memory allocation by the client. Expected to be significantly faster than AES-GCM mode and also faster than a non-DMA `MASKING_NONE` (IPC copy) path.
    *   **Comparison:** The difference between `[time_sgx_aesN]` and `[time_sgx_dmaN]` will show the combined benefit of eliminating AES-GCM and reducing data copies.
    *   **Breakdown Analysis:** For `MASKING_NONE` with DMA (Table 3.2), masking/unmasking components are zero. Data preparation on the client involves `cudaHostAlloc` and `cudaHostGetDevicePointer`. The shared enclave's "IPC Recv" time should be minimal as only pointers are transferred for inputs. GPU execution directly uses client's device memory.

ONNX Model Inference (MobileNetV2)
----------------------------------

**Table 4: ONNX MobileNetV2 - End-to-End Execution Time (seconds)**

+----------------------+----------+-----------------+---------------------------+---------------------------------------+
| Workload             | Native   | Gramine-direct  | Gramine SGX (AES-GCM)     | Gramine SGX (`MASKING_NONE` with DMA) |
+======================+==========+=================+===========================+=======================================+
| MobileNetV2          | [time_n_onnx]| [time_gd_onnx]  | [time_sgx_aes_onnx]       | [time_sgx_dma_onnx]                   |
+----------------------+----------+-----------------+---------------------------+---------------------------------------+

**Table 5: ONNX MobileNetV2 - Calculated Overheads vs Native**

+----------------------+---------------------------+-----------------------------------+-----------------------------------------------+
| Workload             | Gramine-direct Overhead % | Gramine SGX (AES-GCM) Ovhd. %     | Gramine SGX (`MASKING_NONE` with DMA) Ovhd. % |
+======================+===========================+===================================+===============================================+
| MobileNetV2          | [ovhd_gd_onnx]%           | [ovhd_sgx_aes_onnx]%              | [ovhd_sgx_dma_onnx]%                          |
+----------------------+---------------------------+-----------------------------------+-----------------------------------------------+

**Table 6.1: ONNX MobileNetV2 - SGX (AES-GCM Masking) Time Breakdown (ms - Hypothetical)**
*(Existing table structure is fine)*

**Table 6.2: ONNX MobileNetV2 - SGX (`MASKING_NONE` with DMA) Time Breakdown (ms - Hypothetical)**

+---------------------------------+------------+
| Component                       | Time (ms)  |
+=================================+============+
| Client: Input Pinned Alloc & Ptr| [time_c_pinalloc_onnx] |
+---------------------------------+------------+
| Client: IPC Send (Ptr) + Wait   | [time_c_ipc_onnx_dma]   |
+---------------------------------+------------+
| Client: Output Process(No Unmask)| [time_c_ver_nounmask_onnx] |
+---------------------------------+------------+
| Shared Enc: IPC Recv (Ptr)      | [time_s_ipc_recv_ptr_onnx]|
+---------------------------------+------------+
| Shared Enc: DtoD Copy & GPU Exec| [time_s_dtod_gpu_onnx]  | <!-- DtoD + ORT Run -->
+---------------------------------+------------+
| Shared Enc: Output Prep (No Mask)| [time_s_prep_nomask_onnx] |
+---------------------------------+------------+
| **Client End-to-End Total**     | [time_c_total_onnx_dma]|
+---------------------------------+------------+

**Hypothetical Discussion Points (ONNX MobileNetV2):**
    *   (Discussion on Native and Gramine-direct from previous version remains relevant)
    *   **Gramine SGX (AES-GCM) Overhead:** Masking the ~600KB input tensor will be the primary crypto overhead.
    *   **Gramine SGX (`MASKING_NONE` with DMA) Overhead:** Eliminates AES-GCM for the input. The shared enclave performs a Device-to-Device (DtoD) copy from the client's device memory to an enclave-managed device buffer before ONNX Runtime execution. This is faster than HtoD copies and avoids CPU overhead for data handling.
    *   **Comparison:** The difference between SGX modes will quantify the AES-GCM cost plus the benefit of avoiding HtoD copies from enclave CPU memory for the input tensor.
    *   **Breakdown Analysis:** Table 6.2 will show no masking/unmasking time. The "DtoD Copy & GPU Exec" component in the shared enclave will be key.

cuBLAS GEMM (SGEMM)
-------------------

**Table 7: cuBLAS SGEMM - End-to-End Execution Time (seconds)**

+----------------------+----------+-----------------+---------------------------+---------------------------------------+
| Workload (MxN, K)    | Native   | Gramine-direct  | Gramine SGX (AES-GCM)     | Gramine SGX (`MASKING_NONE` with DMA) |
+======================+==========+=================+===========================+=======================================+
| 512x512, K=512       | [time_n_g1]| [time_gd_g1]    | [time_sgx_aes_g1]         | [time_sgx_dma_g1]                     |
+----------------------+----------+-----------------+---------------------------+---------------------------------------+
| 1024x1024, K=1024    | [time_n_g2]| [time_gd_g2]    | [time_sgx_aes_g2]         | [time_sgx_dma_g2]                     |
+----------------------+----------+-----------------+---------------------------+---------------------------------------+
| 2048x2048, K=2048    | [time_n_g3]| [time_gd_g3]    | [time_sgx_aes_g3]         | [time_sgx_dma_g3]                     |
+----------------------+----------+-----------------+---------------------------+---------------------------------------+

**Table 8: cuBLAS SGEMM - Calculated Overheads vs Native**

+----------------------+---------------------------+-----------------------------------+-----------------------------------------------+
| Workload (MxN, K)    | Gramine-direct Overhead % | Gramine SGX (AES-GCM) Ovhd. %     | Gramine SGX (`MASKING_NONE` with DMA) Ovhd. % |
+======================+===========================+===================================+===============================================+
| 512x512, K=512       | [ovhd_gd_g1]%             | [ovhd_sgx_aes_g1]%                | [ovhd_sgx_dma_g1]%                            |
+----------------------+---------------------------+-----------------------------------+-----------------------------------------------+
| 1024x1024, K=1024    | [ovhd_gd_g2]%             | [ovhd_sgx_aes_g2]%                | [ovhd_sgx_dma_g2]%                            |
+----------------------+---------------------------+-----------------------------------+-----------------------------------------------+
| 2048x2048, K=2048    | [ovhd_gd_g3]%             | [ovhd_sgx_aes_g3]%                | [ovhd_sgx_dma_g3]%                            |
+----------------------+---------------------------+-----------------------------------+-----------------------------------------------+

**Table 9.1: cuBLAS SGEMM @ 1024x1024, K=1024 - SGX (AES-GCM Masking) Time Breakdown (ms - Hypothetical)**
*(Existing table structure is fine)*

**Table 9.2: cuBLAS SGEMM @ 1024x1024, K=1024 - SGX (`MASKING_NONE` with DMA) Time Breakdown (ms - Hypothetical)**

+---------------------------------+------------+
| Component                       | Time (ms)  |
+=================================+============+
| Client: Matrix Pinned Alloc&Ptrs| [time_c_pinalloc_gemm] |
+---------------------------------+------------+
| Client: IPC Send (Ptrs) + Wait  | [time_c_ipc_gemm_dma]   |
+---------------------------------+------------+
| Client: Result Verify(No Unmask)| [time_c_ver_nounmask_gemm] |
+---------------------------------+------------+
| Shared Enc: IPC Recv (Ptrs)     | [time_s_ipc_recv_ptrs_gemm]|
+---------------------------------+------------+
| Shared Enc: GPU Exec(Direct DMA)| [time_s_gpu_gemm_dma]   | <!-- cuBLAS uses client pointers -->
+---------------------------------+------------+
| Shared Enc: Result Prep (No Mask)| [time_s_prep_res_gemm]  |
+---------------------------------+------------+
| **Client End-to-End Total**     | [time_c_total_gemm_dma]|
+---------------------------------+------------+

**Hypothetical Discussion Points (cuBLAS GEMM):**
    *   (Discussion on Native and Gramine-direct from previous version remains relevant)
    *   **Gramine SGX (AES-GCM) Overhead:** AES-GCM on large matrices will be substantial.
    *   **Gramine SGX (`MASKING_NONE` with DMA) Overhead:** Similar to VectorAdd, this mode avoids crypto and data copies for input matrices A and B into the shared enclave, using client's device pointers directly. This should yield significant performance improvements.
    *   **Comparison:** The difference between `[time_sgx_aes_gN]` and `[time_sgx_dma_gN]` will highlight the benefits.
    *   **Breakdown Analysis:** Table 9.2 will show minimal client and shared enclave data prep time for inputs. GPU execution time should be similar to native, with IPC and SGX transition costs being the main overheads.

4. Analysis and Bottleneck Identification
=========================================

Impact of Data Transfer Mechanisms and Masking
----------------------------------------------
The shared enclave architecture offers different mechanisms for data handling, each with distinct performance and security trade-offs:

*   **`MASKING_AES_GCM` (Data Copy & Cryptography):**
    *   **Pros:** Highest security for data in transit (IPC) and at rest in the shared enclave's CPU memory.
    *   **Cons:** Significant overhead from AES-GCM encryption/decryption on both client and shared enclave sides. Additionally, data is copied multiple times (client host -> client enclave -> IPC -> shared enclave -> shared enclave host -> GPU device memory, and reverse for results).
    *   **Bottlenecks:** Cryptographic operations, multiple memory copies across protection boundaries, and IPC serialization/deserialization.

*   **`MASKING_NONE` with IPC Copy (Data Copy, No Cryptography):**
    *   **Pros:** Eliminates cryptographic overhead compared to AES-GCM.
    *   **Cons:** Still involves multiple memory copies (client host -> client enclave -> IPC -> shared enclave -> shared enclave host -> GPU device memory). Data is plaintext in IPC and shared enclave CPU memory.
    *   **Bottlenecks:** Multiple memory copies, IPC overhead. Performance is better than AES-GCM but not optimal for large data.

*   **`MASKING_NONE` with Direct GPU DMA (Pointer Passing, Minimal Copies for Inputs):**
    *   **Pros:**
        *   Eliminates cryptographic overhead.
        *   For input data, significantly reduces memory copies. Client allocates pinned host memory, obtains a device pointer, and sends this pointer.
            *   For VectorAdd and GEMM, the shared enclave can use these client device pointers *directly* in CUDA kernels or cuBLAS calls, avoiding any data copy of inputs within the shared enclave.
            *   For ONNX, the current implementation has the shared enclave perform a Device-to-Device (DtoD) copy from the client's device memory region to an enclave-managed device buffer. While this is a copy, a DtoD copy is generally much faster than Host-to-Device (HtoD) copies and avoids involving enclave CPU memory for the bulk data.
        *   This path offers the lowest latency and highest throughput for transferring low-sensitivity input data to the GPU.
    *   **Cons:**
        *   Security: Input data is exposed on the host (pinned memory), PCIe bus, and GPU device memory. This path is only suitable for non-sensitive data.
        *   Complexity: Requires client to manage CUDA pinned memory and device pointers.
    *   **Bottlenecks:** SGX transitions (ECALLs/OCALLs) for IPC and CUDA driver interactions from the shared enclave. For ONNX, the DtoD copy is an additional step, though efficient. Output data is still typically copied back via shared enclave CPU memory.

**Expected Performance Impact of DMA:**
The `MASKING_NONE` with DMA path is expected to be the most performant SGX mode for GPU workloads involving large input datasets. By avoiding both cryptographic operations and the expensive memory copies of input data through enclave CPU memory (and subsequent HtoD copies), it directly addresses major overheads. For operations where the shared enclave can use the client's device pointer directly (VectorAdd, GEMM), the input data transfer overhead from the shared enclave's perspective is minimized to simply receiving the pointer via IPC. For ONNX, the DtoD copy is an efficient way to transfer data already in device memory into a space usable by the ONNX runtime within the shared enclave.

Security Implications of `MASKING_NONE` (DMA or IPC Copy)
---------------------------------------------------------
It is crucial to reiterate that `MASKING_NONE` (whether using DMA or IPC copy for plaintext) means that the GPU data payloads are transferred between enclaves as plaintext and will reside in GPU memory as plaintext. This data is vulnerable to observation by a compromised host OS/hypervisor while on the PCIe bus or in GPU device memory.
This mode should **only** be used if the specific data being processed by the GPU is deemed non-sensitive. The decision must be a careful trade-off between performance and security.

Handling Mixed-Sensitivity Data Workloads
-----------------------------------------

The selective data handling mechanism, allowing a per-request choice between ``MASKING_AES_GCM`` and ``MASKING_NONE`` with DMA, is particularly beneficial for GPU workloads that involve components with varying data sensitivities. This flexibility enables applications to apply strong cryptographic protections where necessary while leveraging performance optimizations for less sensitive data segments.

**1. Data Sensitivity Mapping & Mechanisms**

The choice of mechanism is directly tied to the sensitivity of the data being processed by the GPU:

*   **High Sensitivity Data:**
    *   **Strategy:** Maximum protection for data confidentiality and integrity throughout its lifecycle outside the client enclave's direct control (during IPC, within shared enclave CPU memory, and potentially on the GPU if not end-to-end GPU crypto is available).
    *   **Mechanism:** ``MASKING_AES_GCM``.
    *   **Data Flow:**
        *   *Input:* Client encrypts data with AES-GCM. Ciphertext, IV, and MAC tag are sent via IPC. The shared enclave receives, decrypts the data into its CPU memory, and then copies it to GPU device memory (Host-to-Device).
        *   *Output:* After GPU computation, data is copied from GPU device memory to the shared enclave's CPU memory (Device-to-Host). The shared enclave encrypts this data with AES-GCM, and the resulting ciphertext, IV, and MAC tag are sent back to the client via IPC.
    *   **Overheads:** Includes AES-GCM cryptographic operations (encryption/decryption) on both client and shared enclave sides, multiple memory copies (client host <-> client enclave <-> IPC <-> shared enclave CPU <-> shared enclave GPU), and larger IPC payloads due to ciphertext and GCM tags.

*   **Medium Sensitivity Data:**
    *   **Strategy & Challenges:** The definition of "Medium Sensitivity" can be ambiguous. If the primary concern is data exposure *on the GPU device memory or during PCIe transfer*, then any data that cannot tolerate being plaintext in these locations must be treated as High Sensitivity. The current framework offers two primary paths: full AES-GCM protection or plaintext handling optimized for DMA.
    *   If "Medium Sensitivity" implies that data can be plaintext on the GPU and PCIe bus but requires protection during IPC transit *only*, the current ``MASKING_NONE`` with DMA path does not provide this specific intermediate protection (as DMA implies data is also plaintext on the host for the client's DMA setup). The ``MASKING_NONE`` with IPC copy path (where data is copied into the IPC payload) also transmits plaintext.
    *   **Conclusion for this Framework:**
        1.  If "Medium Sensitivity" data **cannot** be plaintext on the GPU/PCIe bus, it **must be treated as High Sensitivity** and use ``MASKING_AES_GCM``.
        2.  If "Medium Sensitivity" data **can** be plaintext on the GPU/PCIe bus (and the primary concern was, for example, IPC transit if it were over an untrusted network, which is not the case for Gramine's local IPC), then for performance reasons, it would be handled like Low Sensitivity data using ``MASKING_NONE`` with DMA, accepting the associated security implications.

*   **Low Sensitivity Data:**
    *   **Strategy:** Maximum performance by minimizing memory copies and eliminating cryptographic overhead.
    *   **Mechanism:** ``MASKING_NONE`` with client-provided DMA pointers.
    *   **Data Flow:**
        *   *Input:* Client allocates CUDA pinned host memory, generates/places data in it, and obtains corresponding device pointers. These device pointers (and data sizes) are sent via IPC to the shared enclave. The shared enclave uses these device pointers to instruct the GPU to directly access the data (e.g., for CUDA kernels or cuBLAS) or performs an efficient Device-to-Device (DtoD) copy if an intermediate enclave-managed device buffer is necessary (e.g., for current ONNX Runtime integration).
        *   *Output (with DMA output extension):* Client allocates CUDA pinned host memory for results and provides its host pointer (and buffer size) via IPC. The shared enclave, after GPU computation, initiates a Device-to-Host DMA transfer directly from its GPU result buffer to the client's pinned host memory.
    *   **Overheads:** Minimal memory copies. For inputs directly used by kernels, it approaches zero-copy from the shared enclave's perspective. For outputs via DMA, it's one DtoH copy. No cryptographic overhead. IPC payloads are small (pointers and metadata). The main overheads are SGX transitions for IPC/driver calls and client-side pinned memory management.
    *   **Security Implication:** Data is plaintext in client's host memory (pinned buffers), on the PCIe bus during transfer, and in GPU device memory. This mode is only suitable for data where such exposure is acceptable.

**2. Application-Level Strategy for Mixed Workloads**

A single GPU operation request to the shared enclave (e.g., a specific ``vector_add_request_payload_t``) is associated with a single ``masking_level``. It does not support different masking levels for different data elements *within the same request*.

Therefore, to process a task involving components of mixed sensitivities, the client application must:
1.  **Decompose the Task:** Break down the overall workload into a sequence of discrete GPU operation sub-requests.
2.  **Assign Sensitivity:** Determine the appropriate sensitivity level (High or Low, mapping Medium as discussed above) for the data involved in each sub-request.
3.  **Prepare Data & Request:** For each sub-request:
    *   If High Sensitivity: Encrypt data, prepare ``MASKING_AES_GCM`` request.
    *   If Low Sensitivity: Prepare data in pinned memory, obtain relevant pointers, and prepare ``MASKING_NONE`` request with DMA pointers.
4.  **Sequential Invocation:** Send each sub-request to the shared enclave sequentially (or manage dependencies if parallel execution is possible and meaningful for the workload). The client is responsible for assembling the final results from potentially multiple, differently processed sub-requests.

**3. Overall Overhead Considerations for Mixed Workloads**

The total overhead and performance for a mixed-sensitivity workload will be a sum of the overheads incurred by processing each data component according to its chosen path:
*   **High-sensitivity components** will contribute higher latency due to cryptographic operations and multiple data copies associated with ``MASKING_AES_GCM``.
*   **Low-sensitivity components** will benefit from significantly lower latency and higher throughput when processed using ``MASKING_NONE`` with DMA, due to the elimination of cryptographic overhead and minimization of data copies.

By strategically decomposing tasks and applying the appropriate data handling mechanism, applications can achieve a balance: ensuring strong security for sensitive parts of their data while maximizing performance for non-sensitive, bulk data operations on the GPU. The effectiveness of this approach depends on the granularity at which the application can separate its data and operations by sensitivity.

5. Conclusions and Recommendations (Hypothetical)
=================================================

*   **Performance Characteristics Summary:**
    *   Gramine SGX (Shared Enclave) with `MASKING_AES_GCM` provides the highest data protection but incurs significant overhead.
    *   Gramine SGX (Shared Enclave) with `MASKING_NONE` via IPC copy improves performance by removing crypto but still suffers from data copy overhead.
    *   Gramine SGX (Shared Enclave) with `MASKING_NONE` and **DMA** offers the best performance for low-sensitivity data by eliminating crypto and drastically reducing data copy overheads for inputs.
*   **Recommendations for Use:**
    *   Use `MASKING_AES_GCM` when data confidentiality for GPU payloads is paramount.
    *   The `MASKING_NONE` with DMA option is highly recommended for performance-critical operations on non-sensitive data, offering substantial speedups.
    *   **A thorough risk assessment is essential before opting for `MASKING_NONE` (DMA or IPC copy) for any production data.**
    *   Leverage the per-request masking choice to efficiently process mixed-sensitivity workloads.
*   **Potential Future Optimization Areas:**
    *   (Existing points remain relevant)
    *   **ONNX Runtime Direct Device Memory Registration:** Investigate enabling ONNX Runtime (ORT) within the shared enclave to directly use client-provided device memory for input and output tensors, aiming to eliminate Device-to-Device (DtoD) memory copies currently used in the ONNX DMA path. This approach would leverage ORT C APIs such as `CreateTensorWithDataAsOrtValue`, where the `OrtMemoryInfo` parameter can specify that the provided data pointer is resident on a CUDA device.
        *   **Feasibility & Implementation Sketch:**
            1.  **Client-Side:** The client application allocates CUDA device memory for input/output tensors.
            2.  **Memory Sharing with Enclave:** The client provides device pointers (or safer handles like CUDA IPC handles if crossing process boundaries) and tensor metadata (shape, data type, device ID) to the shared enclave.
            3.  **Validation in Enclave:** Within the shared enclave, these pointers/handles must be rigorously validated. This includes verifying they point to accessible and valid device memory regions of the expected size, potentially by interacting with the CUDA driver APIs from within the enclave.
            4.  **ORT Tensor Creation:** The shared enclave uses `CreateTensorWithDataAsOrtValue`, providing the (validated) client's device pointer, tensor shape, data type, and an `OrtMemoryInfo` structure correctly identifying the memory as CUDA device memory (e.g., `OrtCUDAMemoryInfo`).
            5.  **Execution:** The enclave binds these externally-backed `OrtValue`s using `OrtIoBinding` and executes the model with ONNX Runtime.
            6.  **Synchronization:** Explicit CUDA stream and event synchronization (e.g., using ORT's `SynchronizeBoundInputs`/`SynchronizeBoundOutputs` or direct CUDA event mechanisms) is mandatory between the client's operations and the enclave's ORT inference calls.
        *   **Key Challenges & Considerations for Direct ORT Device Memory:**
            *   **Security & Gramine Integration:** Robust validation of external device pointers/handles within the shared enclave is critical. Gramine's memory mapping capabilities must ensure that such external device memory can be safely accessed without compromising SGX EPC protections. Securely importing/validating device memory handles (e.g., CUDA IPC) is preferable to raw pointers.
            *   **ORT API Confirmation:** Thorough testing is needed to confirm `CreateTensorWithDataAsOrtValue` and related APIs behave as expected with externally owned CUDA device pointers in the shared enclave context for both inputs and outputs.
            *   **Lifetime Management:** Clear protocols for managing the lifetime of client-provided device memory are essential; the memory must outlive its use by the enclave. `CreateTensorWithDataAndDeleterAsOrtValue` might be an option if temporary ownership transfer is feasible.
            *   **Synchronization Complexity:** Implementing correct and efficient GPU synchronization across the client-enclave boundary is crucial.
            *   **Performance Trade-offs:** The goal is to ensure that eliminating the DtoD copy results in a net performance gain by keeping the overheads of validation, memory mapping (if any), and synchronization lower than the DtoD copy time.
    *   **Explore DMA for Output Data to Client's Pinned Host Memory:** Investigate optimizing the return of GPU computation results from the shared enclave to the client by using Direct Memory Access (DMA) to client-pre-allocated pinned host memory. This aims to bypass copying output data through the shared enclave's CPU memory and reduce overall latency.
        *   **Implementation Sketch:**
            1.  **Client-Side Allocation:** The client application allocates pinned host memory (e.g., using `cudaHostAlloc`) of sufficient size to hold the expected output data.
            2.  **Buffer Information to Enclave:** The client transmits the host pointer and size of this pinned buffer to the shared enclave via IPC, along with the computation request.
            3.  **Shared Enclave DtoH DMA:** After GPU computation, the shared enclave initiates an asynchronous Device-to-Host memory copy (e.g., `cudaMemcpyAsync`) from the GPU device buffer containing the result directly to the client's provided pinned host memory address.
            4.  **Synchronization:** The shared enclave records a CUDA event after queueing the DtoH copy. The client must synchronize on this event (e.g., via `cudaEventSynchronize` after the event is made available or signaled through IPC) before safely accessing the data in its pinned buffer.
        *   **Key Challenges & Considerations:**
            *   **Security and Validation:** This is the most critical aspect. The shared enclave must rigorously validate the client-provided host pointer and size to ensure it's a legitimate, client-owned pinned memory region. Gramine's mechanisms must ensure that the shared enclave can only write to these specifically designated and validated client memory regions, preventing unauthorized access to other host memory. The risk of the shared enclave writing to arbitrary or malicious host locations due to compromised pointers or internal bugs needs careful mitigation.
            *   **Gramine's Role in Host Memory Access:** Clarify and potentially enhance Gramine's support for allowing an enclave to target specific, client-provided external host memory regions for write operations initiated by CUDA driver calls (like `cudaMemcpyAsync`). This might involve pre-registration of allowable memory regions or other policy enforcement.
            *   **API and Synchronization Complexity:** While CUDA APIs support this, managing the synchronization (events, IPC for event status) across the enclave boundary adds implementation complexity.
            *   **Performance Benefits:** The primary benefit is avoiding a DtoH copy into the shared enclave's memory, followed by an IPC copy to the client. This can significantly reduce latency for large output tensors. The overheads of validation and synchronization must be less than the copy times saved.

6. Raw Data (Placeholder)
=========================
(Unchanged)

*(End of gpu_shared_enclave_analysis.rst)*
