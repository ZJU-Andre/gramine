# Shared Enclave System

## 1. Overview

The Shared Enclave System is designed to act as a secure data relay station for applications running within Gramine SGX enclaves. It enables multiple client enclaves to exchange data with each other through a central server enclave, ensuring that the relay itself is also a trusted SGX application.

Communication between the clients and the server, as well as between the client enclaves themselves (via the server), is secured using:
*   **TLS (Transport Layer Security):** All data in transit is encrypted using TLS 1.2 or higher.
*   **Mutual SGX Remote Attestation (DCAP):** Both the server and the client enclaves perform mutual remote attestation to verify each other's integrity and SGX identity before establishing the application-level communication channel.

## 2. Protocol Definition

### Connection
*   **Transport:** TCP/IP. The server listens on a configurable port (default: 12345).

### Security
*   **Encryption:** TLS 1.2+ for all communication.
*   **Attestation:** Mutual SGX Remote Attestation using DCAP (Datacenter Attestation Primitives).
    *   **Flow:**
        1.  After a successful TLS handshake, the client enclave generates its SGX quote (containing a client-generated nonce or other relevant data in the report data field) and sends it to the server.
        2.  The server enclave verifies the client's quote.
        3.  If the client's quote is valid, the server enclave generates its own SGX quote (containing a hash of its TLS certificate in the report data field) and sends it to the client.
        4.  The client enclave verifies the server's quote, ensuring the server is a genuine SGX enclave and its identity is bound to the current TLS session.
        5.  Only after successful mutual attestation does application-level data exchange begin.

### Message Framing
All application-level messages exchanged after successful TLS and attestation follow this structure:

`| Message Type (1 byte) | Payload Length (4 bytes, Network Byte Order) | Payload (variable) |`

*   **Message Type:** A single byte identifying the type of message.
*   **Payload Length:** A 32-bit unsigned integer in network byte order (big-endian) indicating the size of the subsequent payload in bytes.
*   **Payload:** The actual message data, specific to the message type.

### Message Types

*   **Client to Server:**
    *   `MSG_TYPE_REGISTER_CLIENT (0x01)`
        *   **Payload:** `client_id_string` (UTF-8 string identifying the client)
        *   **Server Response:** `MSG_TYPE_REGISTER_ACK (0x81)` or `MSG_TYPE_REGISTER_NACK (0xE1)` (e.g., if ID is invalid or already in use by an *active* different session).
    *   `MSG_TYPE_SEND_DATA (0x02)`
        *   **Payload:** `recipient_id_len (uint16_t, network byte order)` + `recipient_id_string` + `actual_data_payload`
        *   **Server Response:** `MSG_TYPE_SEND_ACK (0x82)` or `MSG_TYPE_SEND_NACK (0xE2)` (e.g., if recipient is not found/registered).
    *   `MSG_TYPE_POLL_DATA (0x03)`
        *   **Payload:** None.
        *   **Server Response:** `MSG_TYPE_DATA_AVAILABLE (0x83)`, `MSG_TYPE_NO_DATA_AVAILABLE (0xA3)`, or `MSG_TYPE_POLL_NACK (0xE3)`.

*   **Server to Client (Responses):**
    *   `MSG_TYPE_REGISTER_ACK (0x81)`
        *   **Payload:** None.
    *   `MSG_TYPE_REGISTER_NACK (0xE1)`
        *   **Payload:** Optional error message string.
    *   `MSG_TYPE_SEND_ACK (0x82)`
        *   **Payload:** None.
    *   `MSG_TYPE_SEND_NACK (0xE2)`
        *   **Payload:** Optional error message string (e.g., "Recipient not found").
    *   `MSG_TYPE_DATA_AVAILABLE (0x83)`
        *   **Payload:** `sender_id_len (uint16_t, network byte order)` + `sender_id_string` + `actual_data_payload` (The data originally sent by another client).
    *   `MSG_TYPE_NO_DATA_AVAILABLE (0xA3)`
        *   **Payload:** None.
    *   `MSG_TYPE_POLL_NACK (0xE3)`
        *   **Payload:** Optional error message string.
    *   `MSG_TYPE_ERROR (0xFF)`
        *   **Payload:** Optional error message string (for generic errors or unknown message types).

## 3. Components

### Shared Enclave Server (`shared_enclave_server`)
*   **Description:** A C++ application designed to run within a Gramine SGX enclave. It acts as the central relay hub.
*   **Key Features:**
    *   Listens for incoming TCP connections from client enclaves.
    *   Establishes secure TLS sessions with clients.
    *   Performs mutual SGX remote attestation (DCAP) with each client.
    *   Manages client registration using unique client IDs.
    *   Maintains in-memory message queues for each registered client.
    *   Relays data messages from a sender client to a recipient client's queue.
    *   Handles client polls for pending messages.
    *   Uses multiple threads to handle client connections concurrently.

### Example Client Enclave (`example_client`)
*   **Description:** A C++ application, also designed to run within a Gramine SGX enclave, demonstrating how to interact with the Shared Enclave Server.
*   **Interaction Flow:**
    1.  Connects to the Shared Enclave Server via TCP.
    2.  Establishes a TLS session.
    3.  Performs mutual SGX remote attestation (DCAP) with the server.
    4.  Registers itself with the server using its unique client ID.
    5.  Can send data messages to other registered clients via the server.
    6.  Can poll the server for any messages queued for it by other clients.

## 4. Build Instructions

### Prerequisites
*   Gramine SDK (latest stable version recommended).
*   OpenSSL development libraries (e.g., `libssl-dev` on Debian/Ubuntu).
*   A C++ compiler supporting C++11 or later (e.g., `g++`).
*   `make` utility.
*   An SGX-capable machine with SGX drivers and DCAP client libraries installed (for actual SGX execution).

### Steps

1.  **Generate TLS Server Certificate & Key:**
    The server requires a TLS certificate (`server.crt`) and private key (`server.key`). Generate a self-signed pair (for testing):
    ```bash
    openssl req -x509 -newkey rsa:4096 -keyout server.key -out server.crt -days 365 -nodes -subj "/CN=localhost"
    ```
    Place `server.crt` and `server.key` in the directory where the server will be run, or ensure paths in the server manifest are correct.

2.  **Compile Shared Enclave Server:**
    Navigate to the directory containing `server.cpp`, `server.h`, `protocol.h`, and `Makefile`.
    ```bash
    make -f Makefile clean
    make -f Makefile
    ```
    This produces the `server` executable.

3.  **Compile Example Client Enclave:**
    Navigate to the directory containing `client.cpp`, `client.h`, `protocol.h`, and `Makefile_client`.
    ```bash
    make -f Makefile_client clean
    make -f Makefile_client
    mv client example_client # Rename to match manifest
    ```
    This produces the `example_client` executable.

4.  **Generate SGX Signing Key (if needed):**
    To sign the Gramine manifests, an SGX private key is required. If you don't have one, generate a 3072-bit RSA key:
    ```bash
    openssl genrsa -3 -out YOUR_SIGNING_KEY.pem 3072
    ```
    **Important:** Protect this key. For production, use a properly managed signing key.

5.  **Sign Manifests:**
    Use `gramine-sgx-sign` to sign the manifest template files. Replace `YOUR_SIGNING_KEY.pem` with your actual private key file.

    *   **Server Manifest:**
        ```bash
        gramine-sgx-sign \
            --manifest shared_enclave_server.manifest.template \
            --output shared_enclave_server.manifest \
            --key YOUR_SIGNING_KEY.pem
        ```
        This creates `shared_enclave_server.manifest` (signed) and `shared_enclave_server.token` (for SGX driver interaction).

    *   **Client Manifest:**
        ```bash
        gramine-sgx-sign \
            --manifest example_client.manifest.template \
            --output example_client.manifest \
            --key YOUR_SIGNING_KEY.pem
        ```
        This creates `example_client.manifest` (signed) and `example_client.token`.

## 5. Running the System

Ensure all required files (executable, `.manifest` file, `.token` file, and any other files specified in `fs.mounts` like `server.crt`, `server.key`) are in the current directory from which you run `gramine-sgx`.

1.  **Start the Shared Enclave Server:**
    Open a terminal:
    ```bash
    gramine-sgx ./server
    # Or specify a port:
    # gramine-sgx ./server <port_number>
    ```
    The server will start, initialize OpenSSL and its SGX context (if applicable), and begin listening for client connections.

2.  **Start the Example Client Enclave(s):**
    Open one or more new terminals for each client instance.
    ```bash
    gramine-sgx ./example_client <server_ip> <server_port> [client_id]
    ```
    *   `<server_ip>`: IP address of the machine running the server (e.g., `127.0.0.1`).
    *   `<server_port>`: Port the server is listening on (e.g., `12345`).
    *   `[client_id]`: An optional unique string to identify this client (e.g., `ClientA`). Defaults to `default_client_id`.

    Example:
    ```bash
    # Client A
    gramine-sgx ./example_client 127.0.0.1 12345 ClientA
    # Client B (in another terminal)
    gramine-sgx ./example_client 127.0.0.1 12345 ClientB
    ```

## 6. Manifest Configuration

Both `shared_enclave_server.manifest.template` and `example_client.manifest.template` contain crucial settings for running within Gramine SGX. Key options include:

*   `loader.entrypoint = "file:{{ gramine.libos }}"`
*   `libos.entrypoint = "/executable_name"`: Specifies the path to the application binary *inside* the Gramine virtual filesystem (e.g., `/shared_enclave_server` or `/example_client`).
*   `fs.mounts`: Defines the enclave's filesystem view.
    *   Essential for mounting Gramine runtime libraries (`{{ gramine.runtimedir() }}/lib`), system libraries (`/usr/lib`, `/lib`), configuration files (`/etc`), and the application executable itself.
    *   For attestation, access to `/dev/attestation/` (specifically `user_report_data` and `quote`) is critical and must be available through mounts if not intrinsically provided.
*   `sgx.remote_attestation = "dcap"`: Enables DCAP-based remote attestation capabilities for the enclave.
*   `sgx.allowed_files`: A security whitelist.
    *   Must include SGX devices like `/dev/sgx_enclave`, `/dev/sgx_provision`.
    *   For attestation: `/dev/attestation/user_report_data`, `/dev/attestation/quote`.
    *   For networking: `tcp://<ip>:<port>` to allow listening (server) or connecting (client).
    *   Server certificate and key files (`/server.crt`, `/server.key`) if read by the server.
*   `sgx.trusted_files`: Defines the files whose integrity is measured and contributes to the MRENCLAVE value.
    *   This is critical for attestation. It must include the application binary, all dependent shared libraries (including Gramine's LibOS, C standard library, C++ standard library, OpenSSL libraries, etc.), and any other critical read-only data.
    *   Identifying the complete list of trusted files is crucial and can be aided by Gramine's debugging tools (`gramine-sgx-pf-checker`, debug logs).

## 7. Testing

The system's functionality can be verified through several test cases, including:

*   **Successful Data Relay:** Client A sends a message to Client B, Client B polls and receives it.
*   **Client Attestation Failure:** Simulate a client sending an invalid quote to the server; the server should reject it and terminate the connection.
*   **Server Attestation "Verification" by Client:** Client receives and (placeholder) verifies the server's quote.
*   **Error Conditions:**
    *   Sending data to an unregistered/unknown recipient (server should NACK).
    *   Polling for data when no messages are available (server should indicate no data).

Detailed execution steps and expected outputs for these test cases are documented separately (refer to `TEST_CASES.md` if available, or the original development plan). These tests help ensure the core features of secure communication, attestation, and message relay are working correctly.
```
