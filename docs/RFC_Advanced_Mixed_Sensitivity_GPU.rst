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

2.1.1. Manifest Security
------------------------
While the Workload Manifest simplifies workload definition, its own security is crucial. If an attacker (e.g., with access to the client's storage where the manifest might be saved, or able to intercept IPC if the channel is not fully trusted end-to-end by the application for metadata) gains access to a plaintext manifest, they could infer structural details about the application's data processing, potentially correlating segment IDs with intended GPU operations or data characteristics, even if the actual data payloads remain protected.

**Recommended Design Principles for Manifest Security:**

*   **Minimize Sensitive Information by Design:**
    *   Manifest fields should be designed to be as opaque as possible. Instead of direct sensitive strings (e.g., full file paths to sensitive data if the path itself is revealing, or overly descriptive `segment_id` strings like "patient_cancer_scan_segment"), use abstract or coded identifiers (e.g., `segment_id_001`, `op_type_A`).
    *   The client-side helper library would be responsible for resolving these opaque identifiers to concrete data locations or operational parameters *before* data preparation and encryption for high-sensitivity segments, or before DMA setup for low-sensitivity segments. The manifest itself would then carry these less revealing identifiers.

*   **Encryption of the Manifest Content:**
    *   To further protect the structural information, the (potentially opaqued) Workload Manifest content should be encrypted by the client enclave before it is included in, or referenced by, the Batch IPC Request sent to the shared GPU enclave.

**Proposed Encryption Mechanism:**

*   **Method:** AES-GCM is recommended for manifest encryption, providing both confidentiality and integrity.
*   **Key:** A symmetric session key, unique to the client-shared enclave session. This key should be established through a secure handshake protocol between the client enclave and the shared GPU enclave. This process would typically involve remote attestation to verify the identities of both enclaves and establish a shared secret. This key is distinct from the key potentially used for ``MASKING_AES_GCM`` of individual data segments, or could be derived from a master session key.
*   **Workflow:**
    1.  **Manifest Definition:** The client application developer defines the Workload Manifest in plaintext, using opaque identifiers where appropriate.
    2.  **Client Enclave Encryption:** The client enclave's helper library takes the plaintext manifest, serializes it (e.g., to JSON or a binary format), and then encrypts this serialized form using the established session key with AES-GCM. This produces an encrypted manifest ciphertext and a GCM tag.
    3.  **Transmission:**
        *   If the encrypted manifest is relatively small, it can be included directly within the Batch IPC Request.
        *   If it's large, the encrypted manifest could be transmitted once at the beginning of a session (or when it changes) and cached by the shared enclave, referenced subsequently by a handle or hash in Batch IPC Requests. For this proposal, we primarily consider the case where it, or its hash, is part of the batch request for simplicity.
    4.  **Shared Enclave Decryption:** Upon receiving a Batch IPC Request, the shared GPU enclave uses the same session key to decrypt the manifest ciphertext and verify its integrity using the GCM tag. If decryption or verification fails, the entire batch request is rejected.

**Impact on Batch IPC Segment Descriptors:**

*   The `segment_id`s used within the `segment_descriptors` array of the Batch IPC Request would refer to the (now opaque) identifiers from the decrypted manifest. This allows the shared enclave orchestrator to map the IPC segment descriptors back to the detailed entries in the full manifest.
*   The actual sensitive data payloads (e.g., AES-GCM ciphertext for "HIGH" sensitivity segments) or sensitive pointers (e.g., validated DMA pointers for "LOW" sensitivity segments) remain within their respective `segment_descriptors` in the Batch IPC message, handled as previously designed. The manifest encryption primarily protects the *structure and metadata* of the overall workload, not the segment data payloads themselves during IPC (as they have their own protection via ``MASKING_AES_GCM`` or are implicitly accepted as plaintext for DMA).

**Security Considerations:**

*   **Session Key Security:** The entire security of the manifest's confidentiality and integrity relies on the secure establishment, management, and protection of the symmetric session key within both the client and shared enclaves. This key must not be exposed outside these enclaves.
*   **Plaintext Manifest Protection (Client-Side):** The client enclave is responsible for protecting the plaintext manifest *before* it is encrypted. If the manifest is loaded from an untrusted source or constructed with untrusted inputs within the client enclave, its contents could be compromised prior to encryption.
*   **Manifest Replay/Modification:** Using AES-GCM with a unique IV for each encryption (even if the manifest content is identical but part of a new batch request) helps mitigate replay attacks if the manifest is sent per batch. If a manifest is sent once and cached, then the reference handle/hash must be protected against misuse. The GCM tag ensures integrity against modification during transit.

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

8. Shared Enclave Orchestrator: Detailed Design and POC Strategy
===============================================================

This section details the internal design of the Shared Enclave Orchestrator, its core components, error handling, resource management, and a strategy for a Proof-of-Concept (POC) implementation.

8.1. Orchestrator Overview
--------------------------
The Shared Enclave Orchestrator is the central component within the shared GPU enclave responsible for managing the entire lifecycle of a batched GPU workload request. Its key responsibilities include:

*   **Batch Request Management:** Receiving and parsing the batch IPC request from the client enclave.
*   **Manifest Security and Parsing:** Decrypting the (potentially encrypted) Workload Manifest using a pre-established session key and parsing its contents to understand the structure and requirements of the batched segments.
*   **Data Staging (Input):** For each input segment:
    *   **High-Sensitivity:** Decrypting AES-GCM protected data into temporary enclave CPU buffers, then allocating enclave GPU device memory and initiating asynchronous Host-to-Device (HtoD) transfers on an appropriate CUDA stream.
    *   **Low-Sensitivity (DMA):** Validating (within POC limitations) the client-provided device pointer (`src_device_ptr`). If the target GPU operation can use this pointer directly, no data staging within the enclave is needed beyond the validation. If an intermediate enclave-managed buffer is required (e.g., for specific library constraints like the current ONNX setup), an asynchronous Device-to-Device (DtoD) copy is initiated to an enclave-managed buffer on a CUDA stream.
*   **Parallel GPU Dispatch:** Assigning GPU operations (identified by `gpu_operation_id` from the manifest) to dedicated CUDA streams. This allows independent operations to be launched and potentially executed concurrently on the GPU. Future enhancements will include managing dependencies between operations.
*   **Output Handling:** For each output segment:
    *   **High-Sensitivity:** Initiating asynchronous Device-to-Host (DtoH) transfers from the enclave's GPU result buffer to enclave CPU memory. Once complete, encrypting the data using AES-GCM and preparing it for the batch IPC response.
    *   **Low-Sensitivity (DMA):** Validating (within POC limitations) the client-provided host pointer (`dest_host_ptr`) and buffer size. Initiating an asynchronous DtoH transfer directly from the enclave's GPU result buffer to the client's pinned host memory.
*   **Synchronization & Response:** Ensuring all GPU operations and asynchronous memory transfers are complete before assembling and sending the batch response to the client.

8.2. Key Internal Data Structures
-------------------------------

*   **`DecryptedManifest`:**
    *   A C struct or class within the shared enclave that mirrors the structure of the client's Workload Manifest (e.g., an array of `DecryptedSegmentEntry` structs). This is populated after decrypting and parsing the manifest received from the client.
    *   Each `DecryptedSegmentEntry` would contain fields like `segment_id`, `sensitivity_level`, `direction`, `gpu_operation_id`, DMA pointer information (if applicable from the manifest), `data_type_info`, and parsed dependency information (for future use).

*   **`SegmentRuntimeInfo`:**
    *   An array or list of structures, one for each segment described in the `DecryptedManifest`, holding runtime state and resources associated with that segment's processing.
    *   **Fields:**
        *   `segment_id`: Copied from the manifest.
        *   `status`: Current processing status (e.g., PENDING, INPUT_STAGING, GPU_PROCESSING, OUTPUT_HANDLING, COMPLETED, FAILED).
        *   `error_code`: If FAILED, stores an error code.
        *   `masking_level`: Derived from manifest's `sensitivity_level`.
        *   `direction`: From manifest.
        *   `gpu_operation_id`: From manifest.
        *   `assigned_cuda_stream`: Handle to the CUDA stream assigned for this segment's GPU operation and related async copies.
        *   `enclave_cpu_buffer_input`: (For high-sensitivity inputs) Pointer to temporary enclave CPU buffer holding decrypted data.
        *   `enclave_gpu_buffer_input`: (For high-sensitivity inputs or DtoD staging for DMA inputs) Device pointer to enclave-managed GPU memory for input.
        *   `enclave_gpu_buffer_output`: (For all outputs before DtoH) Device pointer to enclave-managed GPU memory for output.
        *   `enclave_cpu_buffer_output`: (For high-sensitivity outputs) Pointer to temporary enclave CPU buffer holding data before encryption.
        *   `client_src_device_ptr`: (For low-sensitivity DMA inputs) Stored from the IPC request.
        *   `client_dest_host_ptr`: (For low-sensitivity DMA outputs) Stored from the IPC request.
        *   `actual_input_size_bytes`: Actual size processed/validated.
        *   `actual_output_size_bytes`: Actual size generated.
        *   Pointers to IV/Tag for AES-GCM segments, potentially stored within the batch response structure being assembled.

*   **`StreamPool` (Conceptual):**
    *   A mechanism to manage a pool of CUDA streams. For the POC, this might be a simple array of `cudaStream_t`.
    *   The orchestrator would assign a stream from this pool to each unique `gpu_operation_id` within a batch request to enable concurrent kernel execution. More advanced implementations might use a hash map from `gpu_operation_id` to `cudaStream_t`.

8.3. Core Orchestration Functions
-------------------------------

*   **`int handle_batch_gpu_request(const BatchIPCRequest* batch_req, BatchIPCResponse* batch_resp)`:**
    *   **Purpose:** Main entry point for handling a batched request.
    *   **Logic:**
        1.  Initialize `batch_resp`.
        2.  Call `decrypt_and_parse_manifest()` using `batch_req->encrypted_manifest_data` (and a session key, see below). If fails, set batch error and return.
        3.  Populate `SegmentRuntimeInfo` array based on the decrypted manifest and segment descriptors in `batch_req`.
        4.  Call `prepare_all_inputs()`. If critical error, set batch error and go to cleanup. Individual segment errors are noted in `SegmentRuntimeInfo`.
        5.  Call `dispatch_all_gpu_operations()`. If critical error, set batch error and go to cleanup. Notes segment errors.
        6.  Call `finalize_all_outputs()`. If critical error, set batch error and go to cleanup. Notes segment errors.
        7.  Perform final synchronization of all used CUDA streams.
        8.  Assemble `batch_resp` based on `SegmentRuntimeInfo` statuses and output data.
        9.  Cleanup resources.
        10. Return overall status.

*   **`int decrypt_and_parse_manifest(const EncryptedManifestData* enc_manifest, DecryptedManifest* dec_manifest)`:**
    *   **Purpose:** Decrypts and validates the Workload Manifest.
    *   **Logic:**
        1.  Retrieve the pre-established symmetric session key for the client session. (POC: May use a hardcoded key initially).
        2.  Perform AES-GCM decryption of `enc_manifest->ciphertext` using the session key and `enc_manifest->iv`, verifying against `enc_manifest->tag`.
        3.  If decryption/verification fails, return error.
        4.  Parse the decrypted plaintext manifest (e.g., JSON or binary) into the `DecryptedManifest` structure. Validate manifest structure.
        5.  Return success/failure.

*   **`int prepare_all_inputs(SegmentRuntimeInfo segments[], int num_segments, const DecryptedManifest* manifest)`:**
    *   **Purpose:** Stages all input data segments for GPU processing.
    *   **Logic:** Iterate through `segments` array:
        1.  If `segment->direction` is INPUT or INPUT_OUTPUT:
            *   If `segment->masking_level == MASKING_AES_GCM`:
                *   Decrypt data from the IPC request's segment descriptor into `segment->enclave_cpu_buffer_input`.
                *   `cudaMalloc` for `segment->enclave_gpu_buffer_input`.
                *   `cudaMemcpyAsync(enclave_gpu_buffer_input, enclave_cpu_buffer_input, ..., HtoD, segment->assigned_cuda_stream)`.
                *   Record CUDA event for this HtoD copy if needed for dependency tracking.
            *   If `segment->masking_level == MASKING_NONE` (DMA input):
                *   Retrieve `client_src_device_ptr` from IPC descriptor.
                *   **CRITICAL VALIDATION (POC: Basic check, TODO: Robust validation):** Perform basic checks on the pointer (e.g., not NULL).
                *   If the `gpu_operation_id` associated with this segment requires an enclave-managed buffer (e.g., current ONNX path):
                    *   `cudaMalloc` for `segment->enclave_gpu_buffer_input`.
                    *   `cudaMemcpyAsync(enclave_gpu_buffer_input, client_src_device_ptr, ..., DtoD, segment->assigned_cuda_stream)`.
                    *   Record CUDA event.
                *   Else (kernel can use client pointer directly):
                    *   `segment->enclave_gpu_buffer_input = client_src_device_ptr;` (conceptually, no actual buffer allocated by enclave here for this specific input segment if pointer is used directly).
        2.  Update `segment->status`. Handle errors per segment.

*   **`int dispatch_all_gpu_operations(SegmentRuntimeInfo segments[], int num_segments, const DecryptedManifest* manifest)`:**
    *   **Purpose:** Launches GPU kernels/operations.
    *   **Logic:** Iterate through `segments` (or unique `gpu_operation_id`s):
        1.  Retrieve `segment->assigned_cuda_stream`.
        2.  Wait for prerequisite events (e.g., input data copy completion event for this stream/operation) using `cudaStreamWaitEvent` if managing dependencies.
        3.  Based on `segment->gpu_operation_id`:
            *   Prepare kernel arguments using appropriate `enclave_gpu_buffer_input` and `enclave_gpu_buffer_output` (allocate output buffer if not already done).
            *   Launch the specific kernel (e.g., `vectorAddKernel<<<...>>>`, `sgemm_kernel<<<...>>>`, `OrtRunAsync(...)`) on `segment->assigned_cuda_stream`.
            *   Record CUDA event after kernel launch for timing or dependency.
        4.  Update `segment->status`. Handle errors per segment.

*   **`int finalize_all_outputs(SegmentRuntimeInfo segments[], int num_segments, BatchIPCResponse* batch_resp)`:**
    *   **Purpose:** Retrieves results from GPU and prepares them for the batch response.
    *   **Logic:** Iterate through `segments`:
        1.  If `segment->direction` is OUTPUT or INPUT_OUTPUT and processing was successful so far:
            *   Wait for GPU operation completion event on `segment->assigned_cuda_stream`.
            *   If `segment->masking_level == MASKING_AES_GCM`:
                *   `cudaMemcpyAsync(segment->enclave_cpu_buffer_output, segment->enclave_gpu_buffer_output, ..., DtoH, segment->assigned_cuda_stream)`.
                *   Record and wait for this DtoH copy event.
                *   Encrypt `enclave_cpu_buffer_output` into the corresponding segment descriptor in `batch_resp`.
            *   If `segment->masking_level == MASKING_NONE` (DMA output):
                *   Retrieve `client_dest_host_ptr` and `dest_buffer_size` from `SegmentRuntimeInfo` (populated from IPC request).
                *   **CRITICAL VALIDATION (POC: Basic check, TODO: Robust validation):** Ensure pointer is not NULL and buffer size is adequate.
                *   `cudaMemcpyAsync((void*)client_dest_host_ptr, segment->enclave_gpu_buffer_output, ..., DtoH, segment->assigned_cuda_stream)`.
                *   Record event. (Client will need to sync on this eventually, or shared enclave syncs all before responding).
                *   Populate `actual_output_data_size_bytes` in `batch_resp` for this segment.
        2.  Update `segment->status`. Handle errors per segment.

8.4. Error Handling Strategy
----------------------------
*   **Per-Segment Errors:** Each function operating on segments (input prep, dispatch, output finalization) will update the `status` and `error_code` fields within the respective `SegmentRuntimeInfo` entry.
*   **Batch-Wide Errors:**
    *   Critical failures (e.g., manifest decryption failure, failure to allocate critical shared resources, unrecoverable CUDA runtime error) will result in setting an overall error status in the `BatchIPCResponse->batch_status`.
    *   If a batch-wide error occurs, processing of further segments may be aborted.
*   **Reporting:** The `BatchIPCResponse` will contain the overall `batch_status` and an array of `segment_responses`, each with its own `status` and potentially error details. This allows the client to understand which parts of a batch succeeded or failed.
*   **POC Simplification:** POC may initially focus on failing the entire batch if any segment encounters a significant error, to simplify error recovery logic.

8.5. Resource Management
------------------------
*   **`DecryptedManifest`:** Allocated per batch request, freed after `handle_batch_gpu_request` completes.
*   **`SegmentRuntimeInfo` Array:** Allocated per batch request, freed after `handle_batch_gpu_request` completes.
*   **Enclave CPU Buffers (`enclave_cpu_buffer_input`, `enclave_cpu_buffer_output`):** Allocated per segment as needed (e.g., for decryption or before encryption). Freed after the data is transferred to/from GPU or encrypted for response.
*   **Enclave GPU Buffers (`enclave_gpu_buffer_input`, `enclave_gpu_buffer_output`):** Allocated per segment. Freed after their contents are processed or copied out (e.g., after kernel launch if input only, after DtoH copy for output).
*   **CUDA Streams (`assigned_cuda_stream`):**
    *   Created from the `StreamPool` (or dynamically) at the start of batch processing for each unique `gpu_operation_id`.
    *   Synchronized before sending the final batch response.
    *   Released/returned to pool after the batch response is sent. (POC: may create/destroy per batch).
*   **CUDA Events:** Created as needed for synchronization (e.g., HtoD/DtoD/kernel/DtoH completion). Freed after use.

8.6. Proof-of-Concept (POC) Implementation Strategy for Orchestrator
--------------------------------------------------------------------

1.  **Basic Batch IPC:** Implement structures for `BatchIPCRequest` and `BatchIPCResponse` supporting a small, fixed number of segment descriptors.
2.  **Simplified Manifest (Embedded or Hardcoded):**
    *   Initially, avoid full manifest parsing. Define a hardcoded `DecryptedManifest` structure within the shared enclave for 2-3 segments representing a simple mixed workload (e.g., one AES-GCM input, one DMA input, one AES-GCM output for a single `gpu_operation_id`).
    *   The client will send a "dummy" batch request that triggers this hardcoded manifest logic.
3.  **Session Key (Hardcoded for POC):** Use a hardcoded AES key for any manifest decryption step (if implemented) and for data segment AES-GCM operations. **Mark with prominent TODO for proper key exchange.**
4.  **Sequential Orchestration (Initial POC):**
    *   Implement `handle_batch_gpu_request` with simplified versions of `prepare_all_inputs`, `dispatch_all_gpu_operations`, and `finalize_all_outputs` that process segments sequentially on a single default CUDA stream. This focuses on getting the data paths correct first.
5.  **Limited GPU Operations:** Support one or two predefined GPU operations. The POC would utilize a set of simple, illustrative CUDA kernels defined in `poc_batch_kernels.cu` (with headers in `poc_batch_kernels.h`). These kernels, such as `launch_poc_copy_kernel_dto_d`, `launch_poc_generate_data_kernel`, and `launch_poc_transform_kernel_float`, would be invoked by the orchestrator based on the `gpu_operation_id` specified in the workload manifest for different segments. For instance:
    *   A `gpu_operation_id` like "OP_SIMPLE_COPY" could map to `launch_poc_copy_kernel_dto_d`.
    *   "OP_INIT_DATA" could map to `launch_poc_generate_data_kernel`.
    *   "OP_SCALE_FLOAT_ARRAY" could map to `launch_poc_transform_kernel_float`.
    These kernels are designed to operate on specific CUDA streams, allowing the POC to demonstrate parallel dispatch and execution.
6.  **Basic DMA Validation (Inputs & Outputs):**
    *   For DMA segments (input `src_device_ptr`, output `dest_host_ptr`), perform only NULL checks.
    *   **Add prominent `TODO: Implement robust security validation for client-provided DMA pointers` comments.**
7.  **Data Handling Paths:**
    *   Implement input path for `MASKING_AES_GCM` (decrypt, HtoD).
    *   Implement input path for `MASKING_NONE` with DMA (use client `src_device_ptr` for DtoD or direct kernel use - POC might initially just do DtoD to an enclave buffer).
    *   Implement output path for `MASKING_AES_GCM` (DtoH, encrypt).
    *   Implement output path for `MASKING_NONE` with DMA (DtoH to client's `dest_host_ptr`).
8.  **Stream Management (Basic):** Once sequential orchestration works, introduce 2-3 CUDA streams. Assign different `gpu_operation_id`s (from the hardcoded manifest) to different streams to demonstrate basic concurrent dispatch capability.
9.  **Error Reporting (Simplified):** Basic error codes for segment failures and overall batch failure.
10. **Resource Cleanup:** Ensure `cudaFree`, `free`, and stream/event destruction for resources allocated within the POC scope.

9. Implementation Guidance for Shared Enclave Orchestrator
===========================================================

This section provides specific guidance for developers tasked with implementing the batch processing logic within the shared GPU enclave, particularly focusing on the `handle_batch_gpu_request` function in `CI-Examples/shared-enclave/src/shared_service.c`.

9.1. Current Implementation Status (Uncompleted Task)
-----------------------------------------------------
The full implementation of the `handle_batch_gpu_request` function, which is the core of the shared enclave orchestrator for batch processing, was **not completed**. This includes the detailed sub-logic for:

*   Input Staging (decrypting/copying/validating input segments).
*   GPU Operation Dispatch (launching CUDA kernels based on manifest instructions).
*   Output Finalization (retrieving results, encrypting/copying to client-specified locations).

The primary reason for this incompletion was persistent tooling issues with the `replace_with_git_merge_diff` tool, which prevented the successful application of the necessary C code changes to the `shared_service.c` file. Multiple attempts were made, but the diffs could not be applied, leaving the batch handling logic in a rudimentary state (basic manifest decryption and dummy response).

Developers taking over this task will need to implement these core components.

9.2. Reference for Proof-of-Concept Strategy
----------------------------------------------
The intended step-by-step plan for a Proof-of-Concept (POC) implementation of the orchestrator is detailed in **Section 8.6 Proof-of-Concept (POC) Implementation Strategy for Orchestrator** of this document. Developers should refer to this section for a phased approach to building the functionality.

9.3. Key Implementation Areas & Hints
-------------------------------------

The following points highlight critical areas and provide guidance for the implementation:

*   **Main Handler Function:**
    *   The central function to implement is `static void handle_batch_gpu_request(const batch_gpu_request_payload_t* req, batch_gpu_response_payload_t* resp)` in `shared_service.c`.
    *   This function will orchestrate the entire lifecycle of a batch request.

*   **Manifest Handling:**
    *   **Decryption:** Implement robust manifest decryption at the beginning of `handle_batch_gpu_request`. The POC currently uses a hardcoded global key (`g_manifest_encryption_key`).
        *   **TODO:** This **must be replaced** with a secure session key mechanism, ideally established via remote attestation and a key exchange protocol between the client and shared enclaves.
    *   **Parsing:** After decryption, the `workload_manifest_t` structure (contained within `req->encrypted_manifest.encrypted_data` after decryption) needs to be parsed to understand the segments and their properties.

*   **Per-Segment Runtime Information:**
    *   Utilize a local structure, such as `SegmentRuntimeInfoPoc` (described in Section 8.2 and used in POC attempts), to track the state of each segment throughout its lifecycle within the batch.
    *   This structure should hold:
        *   `segment_id`, `masking_level`, `direction`, `gpu_operation_id`.
        *   Pointers for enclave-managed host (`enclave_host_buffer_input/output`) and device (`enclave_device_buffer_input/output`) buffers.
        *   The `effective_gpu_input_ptr` which will point to the actual device memory (either client-provided DMA pointer or enclave-allocated device buffer) to be used by the kernel.
        *   `processed_input_data_size` and `actual_output_data_size`.
        *   `status` and `error_code` for that segment.
        *   `assigned_cuda_stream` for asynchronous operations.

*   **Input Staging (Ref: Section 8.6, `prepare_all_inputs` logic):**
    *   Iterate through segments based on the decrypted manifest.
    *   For `MASKING_AES_GCM` inputs:
        1.  Decrypt data from `ipc_segment_descriptor_t->encrypted_data` (using `ipc_segment_descriptor_t->iv` and `ipc_segment_descriptor_t->tag`) into a temporary host buffer (`run_seg->enclave_host_buffer_input`).
        2.  Allocate device memory (`run_seg->enclave_device_buffer_input`) using `cudaMalloc`.
        3.  Asynchronously copy (`cudaMemcpyAsync`) from host to device on the segment's assigned CUDA stream.
        4.  Store the device pointer in `run_seg->effective_gpu_input_ptr`.
    *   For `MASKING_NONE` (DMA) inputs:
        1.  Retrieve `src_device_ptr` from `ipc_segment_descriptor_t`.
        2.  **CRITICAL TODO: SECURITY** - Implement robust validation of `src_device_ptr` (address, size, permissions). This is a significant security concern and may require PAL-level support or extensions. For the POC, basic NULL checks might be a starting point, but this is insufficient for production.
        3.  Set `run_seg->effective_gpu_input_ptr = (void*)ipc_req_seg->src_device_ptr;`.
        4.  `run_seg->processed_input_data_size` comes from `ipc_req_seg->input_data_size`.

*   **GPU Operation Dispatch (Ref: Section 8.6, `dispatch_all_gpu_operations` logic):**
    *   Iterate through `SegmentRuntimeInfoPoc` array. For segments with successful input staging:
    *   Determine the target CUDA kernel based on `run_seg->gpu_operation_id`. The POC kernels are defined in `poc_batch_kernels.cu/.h` (e.g., `launch_poc_copy_kernel_dto_d`, `launch_poc_generate_data_kernel`).
    *   **Output Buffer Allocation:** If the segment produces output, `cudaMalloc` `run_seg->enclave_device_buffer_output`. Determine its size based on manifest hints (`data_size_or_max_output_size`, `client_dma_output_buffer_size`) or input size for copy-like operations.
    *   Launch the selected kernel on the segment's assigned CUDA stream, passing `run_seg->effective_gpu_input_ptr` and `run_seg->enclave_device_buffer_output` as appropriate.
    *   Update `run_seg->actual_output_data_size` based on kernel execution (for POC, this might be fixed or same as input).
    *   Handle kernel launch errors and CUDA errors, updating `run_seg->status`.

*   **Output Finalization (Ref: Section 8.6, `finalize_all_outputs` logic):**
    *   Iterate through `SegmentRuntimeInfoPoc`. For segments processed successfully and marked for output:
    *   Let `ipc_resp_seg = &resp->segments[i];`.
    *   Set `ipc_resp_seg->actual_output_data_size = run_seg->actual_output_data_size;`.
    *   *(Assumed field)* `ipc_resp_seg->masking_level_of_output = run_seg->masking_level;`.
    *   For `MASKING_AES_GCM` outputs:
        1.  `malloc` `run_seg->enclave_host_buffer_output`.
        2.  `cudaMemcpyAsync` DtoH from `run_seg->enclave_device_buffer_output` to host buffer (on segment's stream).
        3.  Synchronize the DtoH copy.
        4.  Encrypt the host buffer into `ipc_resp_seg->encrypted_data` (with `ipc_resp_seg->iv`, `ipc_resp_seg->tag`). Set `ipc_resp_seg->encrypted_data_size`.
    *   For `MASKING_NONE` (DMA) outputs:
        1.  Retrieve `dest_host_ptr` from the corresponding `req->segment_descriptors[i]`.
        2.  **CRITICAL TODO: SECURITY** - Implement robust validation of `dest_host_ptr` (address, size, write permissions).
        3.  `cudaMemcpyAsync` DtoH from `run_seg->enclave_device_buffer_output` directly to `(void*)dest_host_ptr` (on segment's stream).
        4.  `ipc_resp_seg->encrypted_data_size = 0;`.
    *   Handle errors and update `ipc_resp_seg->status`.

*   **CUDA Stream Management:**
    *   Assign a unique `cudaStream_t` to each distinct `gpu_operation_id` (or per segment for simpler POC) to enable potential concurrency.
    *   All asynchronous CUDA calls (`cudaMemcpyAsync`, kernel launches) for a segment's data path should use its assigned stream.
    *   Before sending the final `batch_gpu_response_payload_t`, ensure all operations are complete by synchronizing all relevant CUDA streams (e.g., `cudaDeviceSynchronize()` for simplicity in POC, or individual `cudaStreamSynchronize()`/event-based synchronization for finer control).

*   **Resource Management:**
    *   Meticulously free all `malloc`'d host buffers (`enclave_host_buffer_input`, `enclave_host_buffer_output`).
    *   Meticulously free all `cudaMalloc`'d device buffers (`enclave_device_buffer_input`, `enclave_device_buffer_output`). This is crucial, especially in error paths, to prevent memory leaks.
    *   Manage CUDA stream and event lifecycles if not using the default stream.

*   **Error Handling:**
    *   Follow the strategy in Section 8.4. Each segment response (`ipc_segment_response_t`) should have its `status` field correctly set.
    *   The `resp->overall_batch_status` should reflect if any segment failed or if a batch-wide error occurred.

*   **Iteration Suggestion:**
    *   Begin by implementing the full data path (staging, dummy dispatch, finalization) for a single segment type using the default CUDA stream (0).
    *   Once this is verified, expand to handle multiple segments sequentially.
    *   Finally, introduce multiple CUDA streams to enable parallel execution of independent operations.

9.4. Integration with Existing Code
-----------------------------------
*   The `handle_batch_gpu_request` function is called from `handle_client_session` when `op_type == BATCH_GPU_REQUEST`. Ensure the payload size check and message passing are correct.
*   The IPC structures used (e.g., `batch_gpu_request_payload_t`, `encrypted_manifest_data_t`, `ipc_segment_descriptor_t`, `batch_gpu_response_payload_t`, `ipc_segment_response_t`) must be consistent with their definitions in `CI-Examples/common/shared_service.h`.
*   The POC CUDA kernels are provided in `CI-Examples/shared-enclave/src/poc_batch_kernels.cu` and `CI-Examples/shared-enclave/src/poc_batch_kernels.h`. These need to be compiled and linked with the shared enclave. Ensure the Makefile (`CI-Examples/shared-enclave/Makefile`) is updated to include `poc_batch_kernels.cu` in the CUDA compilation targets and links it into `shared_service.so`.

By following these guidelines and referring to the POC strategy in Section 8.6, developers should be able to complete the implementation of the batch processing orchestrator. The critical areas of DMA pointer validation and secure session key management will require careful design beyond the initial POC.
