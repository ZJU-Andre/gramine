.. SPDX-License-Identifier: LGPL-3.0-or-later
.. Copyright (C) 2023 Intel Corporation

**************************************************
Shared Enclave Architecture with GPU Communication
**************************************************

Introduction
============

This document describes an architecture for Gramine that enables a central service enclave (referred to as the "shared-enclave") to manage resources like GPUs and provide services to multiple client enclaves ("client-enclaves"). A key goal of this architecture is to facilitate tiered data sensitivity handling, allowing applications to balance security requirements with the need to use powerful hardware accelerators like GPUs and shared memory for less sensitive data.

This architecture is particularly useful for scenarios where multiple, potentially untrusted or less trusted, client enclaves need access to a shared, hardware-accelerated service without directly exposing the hardware or sensitive shared data to each client.

Key Components:
---------------
*   **Shared Enclave:** A Gramine enclave that acts as a trusted service provider. It can directly manage resources like GPUs and untrusted shared memory regions.
*   **Client Enclave(s):** Gramine enclaves that consume services provided by the Shared Enclave. They communicate with the Shared Enclave via Gramine's IPC mechanism.
*   **GPU Passthrough:** Allows the Shared Enclave to access host GPU devices.
*   **Untrusted Shared Memory:** A region of memory accessible by both the Shared Enclave and Client Enclaves (and potentially other processes on the host) for exchanging non-sensitive data.
*   **Data Masking:** Cryptographic protection (AES-GCM) for data that is processed by the Shared Enclave and GPU, ensuring its confidentiality and integrity when outside CPU TEE protection.

Manifest Configuration
======================

The behavior and permissions of both Shared and Client enclaves are defined in their respective manifest files. Example templates can be found in ``CI-Examples/shared-enclave/shared-enclave.manifest.template`` and ``CI-Examples/client-enclave/client-enclave.manifest.template``.

Key manifest settings relevant to this architecture include:

*   **`sgx.allowed_files`**:
    *   For the **Shared Enclave** to access GPU devices, these must be listed. For example:
      ::

        sgx.allowed_files = [
          "dev:/dev/nvidia0",
          "dev:/dev/nvidiactl",
          "dev:/dev/nvidia-uvm"
        ]
    *   Client enclaves typically do not need direct access to these GPU device files if they interact with the GPU solely through the Shared Enclave.

*   **`fs.mounts` for Untrusted Shared Memory (`untrusted_shm`)**:
    *   Both **Shared Enclave** and **Client Enclaves** that need to access the shared memory region must mount it. The configuration should be identical for all participating enclaves.
      ::

        fs.mounts = [
          { path = "/untrusted_region", type = "untrusted_shm", uri = "file:/gramine/untrusted_shm/shared_region" },
        ]
    *   The ``path`` specifies the mount point inside the enclave.
    *   The ``type`` must be ``untrusted_shm``.
    *   The ``uri`` (e.g., ``file:/gramine/untrusted_shm/shared_region``) defines the backing host file for this shared memory region. This path must be accessible on the host.

*   **`libos.entrypoint`**:
    *   Defines the main application binary for each enclave. For example:
        *   Shared Enclave: ``libos.entrypoint = "/bin/shared_service"``
        *   Client Enclave: ``libos.entrypoint = "/bin/client_app"``

For detailed manifest syntax, please refer to :ref:`manifest-syntax`.

Inter-Enclave Communication (IPC)
=================================

Communication between Client Enclaves and the Shared Enclave relies on Gramine's built-in IPC mechanism, which is based on PAL (Platform Adaptation Layer) pipes.

*   **Client-Server Model:**
    *   The **Shared Enclave** acts as a server, listening for incoming connections from Client Enclaves. It uses PAL-level functions like ``PalStreamListen()`` on a URI derived from its own VMID and instance ID, and then ``PalStreamAccept()`` to handle new client connections.
    *   **Client Enclaves** initiate connections to the Shared Enclave using Gramine's IPC functions (e.g., ``ipc_connect()``, ``ipc_send_msg_and_get_response()`` from ``libos_ipc.c``), targeting the Shared Enclave's known VMID.

*   **Communication Protocol Example:**
    *   A simple request-response protocol can be established. For instance, the ``shared_service.c`` example (see ``CI-Examples/shared-enclave/src/shared_service.c``) uses ``data_request_t`` and ``data_response_t`` structures:
        .. code-block:: c

            typedef enum {
                STORE_DATA,
                RETRIEVE_DATA
            } operation_type_t;

            typedef enum {
                SENSITIVITY_MEDIUM_GPU, // Data for GPU via Shared Enclave
                SENSITIVITY_LOW_SHM     // Data for untrusted shared memory
            } data_sensitivity_t;

            typedef struct {
                operation_type_t operation;
                data_sensitivity_t sensitivity;
                char path[MAX_PATH_SIZE]; // Identifier or relative path
                uint32_t data_size;
                unsigned char data[MAX_DATA_SIZE]; // Payload
            } data_request_t;

            typedef struct {
                int status; // 0 for success, negative errno for errors
                uint32_t data_size;
                unsigned char data[MAX_DATA_SIZE]; // For retrieved data
            } data_response_t;

    *   Client Enclaves populate ``data_request_t`` and send it to the Shared Enclave. The Shared Enclave processes the request and sends back a ``data_response_t``.

GPU Communication
=================

The Shared Enclave can directly communicate with GPU hardware.

*   **Enabling GPU Access:** This is achieved by listing the relevant GPU device files (e.g., ``/dev/nvidia0``, ``/dev/nvidiactl``) in the ``sgx.allowed_files`` section of the Shared Enclave's manifest.
*   **Memory Mapping:** To interact with the GPU, the Shared Enclave typically needs to map GPU memory into its address space. This is done using the ``mmap()`` syscall on a file descriptor obtained by opening an allowed GPU device file.
    *   Gramine's PAL layer handles this by ensuring that ``mmap()`` calls on ``PAL_TYPE_DEV`` handles (which represent these device files) correctly invoke the host system's ``mmap()`` on the device file descriptor. This functionality is facilitated by `dev_map` in `pal_devices.c` and the modified `generic_emulated_mmap` in `libos_fs_util.c` which diverts to `PalDeviceMap` (conceptually `dev_map`) for device handles. This allows the GPU driver (running on the host) to manage the memory mapping.

Data Sensitivity Handling
=========================

This architecture supports a tiered approach to data sensitivity:

*   **High Sensitivity Data:**
    *   This data is considered most critical (e.g., raw personal data, private keys for client-specific operations).
    *   It should be processed **exclusively within the Client Enclaves**.
    *   It should not be sent to the Shared Enclave or any external resource like the GPU or untrusted shared memory in its raw form.
    *   If such data is used to derive less sensitive outputs that need shared processing, the derivation should happen within the Client Enclave.

*   **Medium Sensitivity Data (Shared Enclave + GPU):**
    *   This data can be processed by the Shared Enclave and potentially offloaded to the GPU for accelerated computation (e.g., model parameters for inference, pre-processed data).
    *   Crucially, when this data is transferred to the Shared Enclave for GPU processing, or when it resides in GPU memory or is in transit over the PCIe bus, it **must be protected using data masking** (see below).
    *   The Shared Enclave is responsible for unmasking the data just before GPU computation and re-masking any sensitive results from the GPU before further storage or transmission.

*   **Low Sensitivity Data (Untrusted Shared Memory):**
    *   Data with low security requirements (e.g., public datasets, non-critical intermediate results, logs for non-sensitive operations) can be placed in the untrusted shared memory region (mounted via ``fs.mounts`` with ``type = "untrusted_shm"``).
    *   This memory is directly accessible by all enclaves that mount it and potentially by other host processes. It offers high performance for data sharing but no confidentiality or integrity guarantees from Gramine/SGX.

Data Masking for GPU Communication
==================================

To protect medium-sensitivity data when it is handled by the Shared Enclave for GPU processing, data masking (encryption and authentication) is essential.

*   **Purpose:** Data masking ensures that sensitive information remains encrypted when it leaves the CPU's TEE protection boundary, such as when it's on the PCIe bus being transferred to the GPU, or while it resides in the GPU's own memory.
*   **Method:** AES-GCM (Galois/Counter Mode) is provided as a robust and widely adopted authenticated encryption algorithm. It provides both confidentiality (encryption) and integrity/authenticity (authentication tag).
*   **Usage:** Gramine LibOS provides helper functions for AES-256-GCM:
    *   ``libos_aes_gcm_encrypt()``
    *   ``libos_aes_gcm_decrypt()``
    *   These functions are declared in ``libos/include/libos_aes_gcm.h``.
*   **Key Management (CRITICAL):**
    *   The security of data masking relies entirely on the secrecy and integrity of the encryption keys.
    *   **Applications are responsible for securely managing these AES-GCM keys.**
    *   Keys should be generated or derived within an enclave that has the authority to handle the specific data. For example, if a Client Enclave sends data to the Shared Enclave for processing, the Client Enclave might encrypt it with a key shared (securely) with the Shared Enclave, or the Shared Enclave itself might use its own keys if it's the designated data protector for that stage.
    *   Consider using SGX sealing mechanisms (e.g., ``mbedtls_sgx_seal_keys()`` if using mbedTLS, or PAL-level sealing) to protect AES keys at rest within the enclave that manages them.
    *   The provided ``libos_aes_gcm_encrypt/decrypt`` functions require the caller to provide the key.
*   **Initialization Vector (IV) Management:**
    *   An IV must be unique for every encryption operation performed with the same key. Reusing an IV with the same key completely compromises GCM's security.
    *   It is recommended to generate IVs using a cryptographically secure random number generator (e.g., ``PalRandomBitsRead()`` or mbedTLS's RNG facilities).
    *   The IV does not need to be secret and can be transmitted alongside the ciphertext.

Workflow Summary
================

1.  **Client Enclave:**
    a.  Prepares data.
    b.  **High-sensitivity data:** Processed locally.
    c.  **Medium-sensitivity data destined for GPU (via Shared Enclave):**
        i.  Client may send it raw to the Shared Enclave if the IPC channel is considered secure enough for this hop and the Shared Enclave is trusted to apply masking. Alternatively, the client could pre-mask it if it holds the appropriate key.
        ii. Sends a request (e.g., ``data_request_t``) to the Shared Enclave via IPC.
    d.  **Low-sensitivity data:** Can be directly written to/read from the configured ``untrusted_shm`` path (e.g., ``/untrusted_region``).

2.  **Shared Enclave:**
    a.  Receives the request from a Client Enclave.
    b.  If the request involves GPU processing of medium-sensitivity data:
        i.  If data arrived raw from client: Calls ``libos_aes_gcm_encrypt()`` to mask the data using a key it securely manages.
        ii. Transfers the masked data to GPU memory (e.g., via CUDA memcpy to memory mapped using ``mmap`` on the GPU device FD).
        iii. Initiates GPU computation (e.g., CUDA kernel launch).
        iv. If sensitive data is read back from the GPU: This data is also masked by the GPU driver/runtime or should be immediately masked by the Shared Enclave. The Shared Enclave then calls ``libos_aes_gcm_decrypt()`` to get the plaintext results for further processing or before sending back to the client (potentially re-masked with a client-specific key or sent raw if the IPC channel is trusted for that result).
    c.  If the request involves low-sensitivity data in ``untrusted_shm``, it accesses the path directly.
    d.  Sends a response (e.g., ``data_response_t``) back to the Client Enclave via IPC.

Security Considerations
=======================

*   **Key Management:** As highlighted, the security of the data masking feature (AES-GCM) is critically dependent on the secure management of encryption keys. Keys should be protected within the enclaves that use them (e.g., using SGX sealing).
*   **Trust Model of Shared Enclave:** Users of this architecture must inherently trust the code and integrity of the Shared Enclave. A compromised Shared Enclave could potentially misuse data it has access to, even if masked during transit to the GPU.
*   **Untrusted Shared Memory:** Data placed in the ``untrusted_shm`` region is not protected by Gramine/SGX confidentiality or integrity mechanisms. It is accessible to any process on the host that can access the underlying host file. Use this only for truly non-sensitive data.
*   **GPU Hardware and Drivers:** Vulnerabilities in GPU hardware, firmware, or host-side drivers are outside the scope of Gramine's TEE protections. While data masking protects data in transit and at rest on the GPU (from an OS perspective), sophisticated hardware attacks or compromised drivers could still pose a risk.
*   **IPC Channel Security:** While Gramine's IPC is designed for inter-enclave communication, the data itself is traversing untrusted OS components (though typically via local pipes). For highly sensitive data exchange between client and shared enclaves, consider if end-to-end encryption over the IPC channel is necessary, in addition to the masking applied for GPU offload.
