######################################################################
Proposal: Automated and Parallel Handling of Mixed-Sensitivity GPU Workloads
######################################################################

.. contents::
   :local:
   :depth: 2

1. Introduction & Motivation
============================

Problem
-------
The current Gramine shared GPU enclave architecture provides mechanisms for handling data of different sensitivities (``MASKING_AES_GCM`` for high-sensitivity, ``MASKING_NONE`` with DMA for low-sensitivity). However, complex workloads that involve multiple data segments with varying sensitivities, potentially processed by different GPU operations, face limitations:

1.  **Manual Decomposition and Orchestration:** Client applications must manually decompose their workload into individual segments, decide the sensitivity and appropriate data handling mechanism for each, and manage the sequential submission of these segments to the shared enclave. This increases client-side complexity and is error-prone.
2.  **Sequential Processing:** Each segment is typically processed sequentially by the shared enclave. This underutilizes the GPU's potential for parallel execution, especially when segments could be processed independently by concurrent GPU kernels.
3.  **IPC Overhead:** Sending multiple small, independent requests incurs higher cumulative IPC overhead compared to batching.

Goal
----
This proposal outlines a conceptual design to automate the handling of mixed-sensitivity GPU workloads and enable parallel processing of independent data segments within the shared GPU enclave. The aim is to:

*   Simplify client application development by abstracting data preparation and request generation.
*   Improve performance by enabling parallel GPU operations on independent data segments using CUDA Streams.
*   Reduce IPC overhead by batching multiple segment requests.

Illustrative Use Case: Medical Image Analysis Model
---------------------------------------------------
Consider a medical imaging AI model that processes:
*   **Sensitive Patient Image Data:** Requires strong confidentiality (e.g., handled with ``MASKING_AES_GCM``). This might involve a large image processed by a primary inference kernel.
*   **Non-Sensitive Model Configuration/Metadata:** Can be handled as plaintext (e.g., via ``MASKING_NONE`` with DMA). This might be smaller configuration data used by the same or an auxiliary GPU kernel.
*   **Non-Sensitive Calibration Data:** Another plaintext segment used by a separate calibration kernel that can run concurrently with the main inference.

Currently, the client would manually manage the encryption of the image, prepare separate DMA buffers for metadata and calibration data, and send multiple requests, likely processed sequentially. The proposed design aims to allow the client to define these segments and their properties, and have the system automatically handle their preparation, batching, parallel execution on the GPU (where dependencies allow), and result aggregation.

2. Proposed Conceptual Design
===========================

The proposed design introduces a workload manifest, client-side automation, batch IPC, and enhanced shared enclave orchestration.

2.1. Workload Manifest
----------------------
The Workload Manifest is a client-defined structure (e.g., a JSON or structured data file/buffer) that describes all data segments involved in a complex GPU task. It serves as a blueprint for both the client-side helper library and the shared enclave orchestrator.

**Key Fields per Data Segment Entry:**

*   `segment_id`: A unique identifier for the data segment (e.g., "patient_scan_slice1", "model_config_params", "calibration_coeffs").
*   `data_location_client`: Path to the data on the client's filesystem or a memory buffer identifier.
*   `sensitivity_level`: Specifies the data sensitivity (e.g., "HIGH", "LOW").
*   `direction`: Indicates if the segment is "INPUT", "OUTPUT", or "INPUT_OUTPUT".
*   `gpu_operation_id`: An identifier for the specific GPU kernel/operation this segment is intended for (e.g., "INFERENCE_KERNEL_MAIN", "PREPROC_KERNEL_AUX", "CALIBRATION_ADJUST"). This allows the shared enclave to map segments to appropriate GPU streams and kernels.
*   `data_type_info`: (Optional) Information about data type, dimensions, expected size, which can aid validation.
*   **For INPUT segments (if `sensitivity_level` is "LOW" and DMA is intended):**
    *   `client_dma_input_pinned_host_ptr_var`: Name of a client-side variable that will hold the host pointer to pinned memory.
    *   `client_dma_input_device_ptr_var`: Name of a client-side variable that will hold the device pointer.
*   **For OUTPUT segments (if `sensitivity_level` is "LOW" and DMA output is intended):**
    *   `client_dma_output_pinned_host_ptr_var`: Name of a client-side variable holding the host pointer to the pre-allocated pinned output buffer.
    *   `client_dma_output_buffer_size_var`: Name of a client-side variable holding the size of this output buffer.
*   `dependencies`: (Future Work) A list of `segment_id`s that this segment's processing depends on, enabling a dependency graph for GPU operations.

2.2. Client-Side Automated Data Preparation
-------------------------------------------
A client-side helper library will automate data preparation and batch request generation based on the Workload Manifest.

*   **Manifest Parsing:** The library reads the manifest.
*   **Automated Data Handling:**
    *   For segments marked "HIGH" sensitivity:
        *   Reads data from `data_location_client`.
        *   Automatically performs AES-GCM encryption.
        *   Prepares the data structure for batched IPC (inline encrypted data, IV, tag).
    *   For segments marked "LOW" sensitivity (intended for DMA):
        *   Reads data from `data_location_client`.
        *   Allocates pinned host memory (`cudaHostAlloc`).
        *   Copies data to pinned memory.
        *   Gets the corresponding device pointer (`cudaHostGetDevicePointer`).
        *   Stores these pointers in variables named by `client_dma_input_pinned_host_ptr_var` and `client_dma_input_device_ptr_var`.
        *   Prepares the data structure for batched IPC (device pointer, size).
    *   For "LOW" sensitivity OUTPUT segments intended for DMA:
        *   Allocates pinned host memory for results based on expected size (either from manifest or a default).
        *   Stores pointer and size in variables named by `client_dma_output_pinned_host_ptr_var` and `client_dma_output_buffer_size_var`.
*   **Batch Request Generation:** Assembles a single batch IPC request containing descriptors for all processed segments.

2.3. Batch IPC Communication
----------------------------
To reduce IPC overhead, multiple segment operations are batched into a single request/response pair.

*   **Batch Request Structure:**
    *   `manifest_id_or_hash`: A reference to the Workload Manifest (e.g., a hash or a unique ID if the manifest was pre-registered with the shared enclave).
    *   `num_segments`: Number of segments in this batch.
    *   `segment_descriptors[]`: An array of descriptors, one for each segment. Each descriptor includes:
        *   `segment_id`: From the manifest.
        *   `masking_level`: Determined by the client library (e.g., `MASKING_AES_GCM`, `MASKING_NONE`).
        *   **If `MASKING_AES_GCM` (INPUT):**
            *   `iv`, `tag`, `encrypted_data_payload` (or offset/length if data is appended).
        *   **If `MASKING_NONE` with DMA (INPUT):**
            *   `src_device_ptr`, `data_size_bytes`.
        *   **If `MASKING_NONE` with DMA (OUTPUT):**
            *   `dest_host_ptr`, `dest_buffer_size_bytes`.
        *   `gpu_operation_id`: To map to the correct GPU operation in the shared enclave.

*   **Batch Response Structure:**
    *   `batch_status`: Overall status of the batch processing.
    *   `num_segments`: Number of segment responses.
    *   `segment_responses[]`: An array of responses, corresponding to input segments. Each includes:
        *   `segment_id`.
        *   `status`: Status for this specific segment's processing.
        *   **If `MASKING_AES_GCM` (OUTPUT):**
            *   `iv`, `tag`, `encrypted_data_payload`.
        *   **If `MASKING_NONE` with DMA (OUTPUT):**
            *   `actual_output_data_size_bytes` (data is already in client's pinned buffer).
        *   Error information if applicable.

2.4. Shared Enclave Orchestration
---------------------------------
The shared enclave receives the batch request and orchestrates parallel execution.

*   **Batch Parsing:** Parses the batch request and segment descriptors. Validates against a cached/referenced manifest if necessary.
*   **Data Staging & CUDA Stream Allocation:**
    *   For each `gpu_operation_id` encountered, a dedicated CUDA stream is created if not already available for this batch.
    *   For each segment:
        *   **High-Sensitivity Input (`MASKING_AES_GCM`):**
            1.  Decrypts data into enclave CPU memory.
            2.  Allocates enclave GPU device memory.
            3.  Copies decrypted data from CPU to GPU (HtoD) asynchronously on the segment's assigned CUDA stream.
        *   **Low-Sensitivity DMA Input (`MASKING_NONE`):**
            1.  Validates the client-provided `src_device_ptr` (CRITICAL: placeholder for actual robust validation mechanism).
            2.  If the target GPU kernel can directly use this pointer (e.g., custom kernels, VectorAdd, GEMM), it's used as is.
            3.  If an intermediate enclave-managed buffer is needed (e.g., current ONNX design), performs an asynchronous DtoD copy from `src_device_ptr` to an enclave-managed device buffer on the segment's assigned CUDA stream.
*   **Parallel GPU Operation Dispatch:**
    *   GPU operations (kernels, library calls like cuBLAS, ONNX Run) associated with different `gpu_operation_id`s are launched on their respective CUDA streams. This allows independent operations to execute concurrently on the GPU.
    *   Dependencies between operations (if defined in the manifest - future work) would require stream synchronization primitives (e.g., `cudaStreamWaitEvent`).
*   **Parallel Output Handling:**
    *   For each segment:
        *   **High-Sensitivity Output (`MASKING_AES_GCM`):**
            1.  Copies data from enclave's GPU device memory to enclave CPU memory (DtoH) asynchronously on its stream.
            2.  Once DtoH completes (stream synchronization for this copy), encrypts data with AES-GCM.
            3.  Places encrypted data, IV, tag into the corresponding segment response.
        *   **Low-Sensitivity DMA Output (`MASKING_NONE`):**
            1.  Validates client-provided `dest_host_ptr` and `dest_buffer_size_bytes`.
            2.  Performs an asynchronous DtoH copy from the enclave's GPU result buffer directly to the client's `dest_host_ptr` on its stream.
            3.  Records the `actual_output_data_size_bytes` in the segment response.
*   **Synchronization and Response:**
    *   Before sending the batch response, the shared enclave synchronizes all involved CUDA streams (`cudaStreamSynchronize` or per-stream event synchronization) to ensure all GPU operations and data transfers (including DtoH DMA to client memory) are complete.
    *   Assembles and sends the batch response.

2.5. GPU Kernel Considerations
------------------------------
GPU kernels intended for use in this framework should be designed to accept device pointers for their respective input and output data segments. This is standard practice for CUDA programming and allows the orchestration layer to manage memory and pass the correct device pointers to the kernels when they are launched on their assigned streams.

3. Data Flow Example (Illustrative)
===================================

Consider a batch with two input segments for two different GPU operations:
*   **Seg1 (Image):** High-sensitivity, `gpu_operation_id="OP1_INFERENCE"`
*   **Seg2 (Metadata):** Low-sensitivity (DMA), `gpu_operation_id="OP2_AUX_CONFIG"`
And one output segment for OP1:
*   **Seg3 (Results):** High-sensitivity, `gpu_operation_id="OP1_INFERENCE"` (output of OP1)

**Flow:**

1.  **Client:**
    *   Helper library parses manifest.
    *   Seg1: Encrypts image -> `enc_img_data`.
    *   Seg2: Prepares metadata in pinned host memory -> `pinned_meta_host_ptr`, gets `meta_dev_ptr`.
    *   Assembles Batch IPC:
        *   `{ manifest_ref, num_segments=3,`
        *   `  descriptors: [`
        *   `    { seg1, MASKING_AES_GCM, data=(enc_img_data,iv,tag), op_id="OP1_INFERENCE" },`
        *   `    { seg2, MASKING_NONE, dma_ptr=meta_dev_ptr, size=..., op_id="OP2_AUX_CONFIG" },`
        *   `    { seg3, MASKING_AES_GCM, op_id="OP1_INFERENCE", direction=OUTPUT } // Output placeholder`
        *   `  ]`
        *   `}`

2.  **Shared Enclave (Batch Request Received):**
    *   Orchestrator:
        *   Creates `cudaStreamOp1`, `cudaStreamOp2`.
        *   **Seg1 (Image):**
            *   Decrypts `enc_img_data` to `plain_img_cpu` (enclave CPU).
            *   `cudaMalloc` `img_dev_enclave_buf`.
            *   `cudaMemcpyAsync(img_dev_enclave_buf, plain_img_cpu, ..., HtoD, cudaStreamOp1)`.
        *   **Seg2 (Metadata):**
            *   Validates `meta_dev_ptr`.
            *   (If needed for OP2_AUX_CONFIG, DtoD copy to enclave buffer on `cudaStreamOp2`, else use `meta_dev_ptr` directly).
        *   **Seg3 (Results):**
            *   `cudaMalloc` `results_dev_enclave_buf` for OP1 output.
    *   GPU Dispatch:
        *   `kernel_OP1<<<..., cudaStreamOp1>>>(img_dev_enclave_buf, results_dev_enclave_buf, ...)`.
        *   `kernel_OP2<<<..., cudaStreamOp2>>>(meta_dev_ptr_or_enclave_copy, ...)`.
    *   Output Handling & Sync:
        *   `cudaMemcpyAsync(plain_results_cpu, results_dev_enclave_buf, ..., DtoH, cudaStreamOp1)`.
        *   `cudaStreamSynchronize(cudaStreamOp1)`.
        *   `cudaStreamSynchronize(cudaStreamOp2)`.
        *   Encrypt `plain_results_cpu` -> `enc_results_data`.
    *   Assembles Batch IPC Response:
        *   `{ batch_status=OK, num_segments=1,`
        *   `  responses: [ { seg3, status=OK, data=(enc_results_data,iv,tag) } ]`
        *   `}`
        *   (Responses for input-only segments might just indicate status).

4. Expected Benefits
====================

*   **Reduced Manual Client Effort:** Clients define workloads declaratively via the manifest. The helper library and shared enclave handle the complexities of data preparation, encryption, DMA setup, and IPC batching.
*   **Performance Improvement:**
    *   **Parallelism:** Concurrent execution of independent GPU operations (mapped to different `gpu_operation_id`s) on separate CUDA streams can significantly improve GPU utilization and reduce overall latency for complex tasks.
    *   **Reduced IPC Overhead:** Batching multiple segment operations into a single IPC request/response cycle reduces the number of SGX transitions and fixed IPC costs.
    *   **Optimized Data Transfer:** Leverages DMA for low-sensitivity data, minimizing copies, and uses efficient DtoD copies within the enclave where necessary.

5. Overheads and Costs
======================
While aiming for performance, this design introduces its own overheads:

*   **Manifest Parsing and Management:** Overhead on client and potentially shared enclave.
*   **Client-Side Helper Library:** Computation for data preparation (though intended to be less than manual effort).
*   **Batch IPC Serialization/Deserialization:** Handling potentially larger and more complex batched messages.
*   **Shared Enclave Orchestration Logic:** Parsing, stream management, conditional data handling (decrypt, copy). More complex than single request processing.
*   **CUDA Stream Management:** Creation, synchronization (though generally lightweight).
*   **Memory Usage:** Pinned host memory on the client; potentially more enclave device/CPU memory for staging and parallel operations.

These are expected to be outweighed by gains from parallelism and reduced IPC for suitable workloads.

6. Critical Challenges & Future Work
====================================

*   **DMA Pointer Security & Validation (CRITICAL):**
    *   Ensuring client-provided DMA pointers (both device source and host destination) are valid and cannot be used to compromise the shared enclave or host system is paramount. This requires robust validation mechanisms, potentially involving Gramine PAL extensions for secure memory region registration or verification.
*   **Error Handling and Rollback:** Managing errors for individual segments within a batch and deciding on overall batch success/failure or partial results.
*   **Dependency Management:** Implementing a full dependency graph (`dependencies` field in manifest) to allow the orchestrator to correctly sequence operations that depend on each other, while still parallelizing independent branches. This would involve more sophisticated CUDA event management.
*   **Dynamic Buffer Sizing for DMA Output:** Accurately predicting output sizes for client-preallocated DMA buffers can be challenging. Mechanisms for handling size mismatches or two-phase (metadata then data) transfers might be needed for some use cases.
*   **Manifest Schema and Versioning:** Defining a stable and extensible manifest format.
*   **Granularity of `gpu_operation_id`:** Finding the right balance for defining operations to maximize parallelism without excessive fragmentation.

7. Conclusion
=============
The proposed design for automated and parallel handling of mixed-sensitivity GPU workloads offers a path to significantly enhance the usability and performance of the Gramine shared GPU enclave system for complex applications. By introducing a workload manifest and enabling batched, stream-based parallel processing in the shared enclave, it aims to reduce client-side burden and better utilize GPU resources. Addressing the critical security challenges associated with DMA pointer validation will be key to a successful and secure implementation. This approach has the potential to make secure GPU processing in enclaves more accessible and efficient for real-world, multifaceted workloads.
