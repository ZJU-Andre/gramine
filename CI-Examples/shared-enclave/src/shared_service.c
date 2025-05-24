#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h> // For sleep, close
#include <sys/stat.h> // For mkdir
#include <math.h>   // For fabs, if needed for float comparisons (not directly here but good for context)

// Gramine includes
#include "libos_ipc.h"
#include "libos_types.h"
#include "pal.h"
#include "libos_aes_gcm.h" // For AES-GCM operations

// Example-specific includes
#include "shared_service.h" // Common definitions for client/server (from ../common/)
#include "vector_add.h"     // For launch_vector_add_cuda

#define UNTRUSTED_SHM_PATH_PREFIX "/untrusted_region"

// TODO: Secure Key Management. For this example, the key is hardcoded.
// In a real application, this key must be protected (e.g., sealed by SGX or derived via attestation).
static const unsigned char g_shared_enclave_aes_key[GCM_KEY_SIZE_BYTES] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f
};

// Helper to generate random IVs - TODO: Use a CSPRNG for production
static void generate_iv_for_response(unsigned char* iv, size_t iv_len) {
    // For testing, simple predictable IVs might be okay, but for production, random is better.
    // This should use a cryptographically secure random number generator.
    // PalRandomBitsRead(iv, iv_len); // Example if PalRandomBitsRead is available and suitable
    for (size_t i = 0; i < iv_len; ++i) {
        iv[i] = (unsigned char)(rand() % 256); // Replace with proper RNG
    }
}

// --- Placeholder GPU Functions (Legacy, can be removed if not used by legacy requests) ---
int legacy_store_to_gpu(const char* id, const unsigned char* data, uint32_t size) {
    printf("SHARED_SERVICE_LOG: Legacy Storing to GPU (ID: %s, Size: %u) - Placeholder\n", id, size);
    if (!id || !data) return -EINVAL;
    // Assuming data is already in a suitable format if this were real
    return 0; // Success
}

int legacy_retrieve_from_gpu(const char* id, unsigned char* buffer, uint32_t buffer_max_size, uint32_t* size_read) {
    printf("SHARED_SERVICE_LOG: Legacy Retrieving from GPU (ID: %s) - Placeholder\n", id);
    if (!id || !buffer || !size_read) return -EINVAL;
    const char* dummy_data = "SampleLegacyGPUData";
    uint32_t dummy_size = strlen(dummy_data);
    if (dummy_size > buffer_max_size) dummy_size = buffer_max_size;
    memcpy(buffer, dummy_data, dummy_size);
    *size_read = dummy_size;
    printf("SHARED_SERVICE_LOG: Retrieved %u bytes from GPU (ID: %s)\n", *size_read, id);
    return 0; // Success
}

// --- Request Handlers ---
// Handler for legacy STORE_DATA operations
void handle_legacy_store_request(const legacy_data_request_t* req, legacy_data_response_t* resp) {
    resp->data_size = 0; 

    if (req->sensitivity == SENSITIVITY_MEDIUM_GPU) {
        printf("SHARED_SERVICE_LOG: Handling LEGACY STORE request for GPU (Path/ID: %s, Size: %u)\n", req->path, req->data_size);
        resp->status = legacy_store_to_gpu(req->path, req->data, req->data_size);
    } else if (req->sensitivity == SENSITIVITY_LOW_SHM) {
        char full_path[MAX_LEGACY_PATH_SIZE + sizeof(UNTRUSTED_SHM_PATH_PREFIX) + 1];
        snprintf(full_path, sizeof(full_path), "%s/%s", UNTRUSTED_SHM_PATH_PREFIX, req->path);
        printf("SHARED_SERVICE_LOG: Handling LEGACY STORE request for SHM (Full Path: %s, Size: %u)\n", full_path, req->data_size);

        // ... (SHM store logic as before, ensure it uses req->data_size)
        char* last_slash = strrchr(full_path, '/');
        if (last_slash) {
            *last_slash = '\0'; 
            if (mkdir(full_path, 0775) != 0 && errno != EEXIST) {
                perror("SHARED_SERVICE_ERROR: mkdir failed for SHM path");
                resp->status = -errno;
                *last_slash = '/'; 
                return;
            }
            *last_slash = '/'; 
        }
        FILE* fp = fopen(full_path, "wb");
        if (!fp) {
            perror("SHARED_SERVICE_ERROR: fopen for legacy SHM store failed");
            resp->status = -errno;
            return;
        }
        size_t written = fwrite(req->data, 1, req->data_size, fp);
        if (written != req->data_size) {
            perror("SHARED_SERVICE_ERROR: fwrite for legacy SHM store failed");
            resp->status = ferror(fp) ? -EIO : -ENOSPC;
            fclose(fp);
            return;
        }
        fclose(fp);
        printf("SHARED_SERVICE_LOG: Successfully stored %zu bytes to legacy SHM path %s\n", written, full_path);
        resp->status = 0;
    } else {
        fprintf(stderr, "SHARED_SERVICE_ERROR: Unknown sensitivity type for LEGACY STORE: %d\n", req->sensitivity);
        resp->status = -EINVAL;
    }
}

// Handler for legacy RETRIEVE_DATA operations
void handle_legacy_retrieve_request(const legacy_data_request_t* req, legacy_data_response_t* resp) {
    if (req->sensitivity == SENSITIVITY_MEDIUM_GPU) {
        printf("SHARED_SERVICE_LOG: Handling LEGACY RETRIEVE request for GPU (Path/ID: %s)\n", req->path);
        resp->status = legacy_retrieve_from_gpu(req->path, resp->data, sizeof(resp->data), &resp->data_size);
    } else if (req->sensitivity == SENSITIVITY_LOW_SHM) {
        char full_path[MAX_LEGACY_PATH_SIZE + sizeof(UNTRUSTED_SHM_PATH_PREFIX) + 1];
        snprintf(full_path, sizeof(full_path), "%s/%s", UNTRUSTED_SHM_PATH_PREFIX, req->path);
        printf("SHARED_SERVICE_LOG: Handling LEGACY RETRIEVE request for SHM (Full Path: %s)\n", full_path);

        FILE* fp = fopen(full_path, "rb");
        if (!fp) {
            perror("SHARED_SERVICE_ERROR: fopen for legacy SHM retrieve failed");
            resp->status = -errno;
            resp->data_size = 0;
            return;
        }
        size_t read_bytes = fread(resp->data, 1, sizeof(resp->data), fp); // Read up to max buffer size
        if (ferror(fp)) {
            perror("SHARED_SERVICE_ERROR: fread for legacy SHM retrieve failed");
            resp->status = -EIO;
            resp->data_size = 0;
            fclose(fp);
            return;
        }
        fclose(fp);
        resp->data_size = read_bytes;
        printf("SHARED_SERVICE_LOG: Successfully retrieved %u bytes from legacy SHM path %s\n", resp->data_size, full_path);
        resp->status = 0;
    } else {
        fprintf(stderr, "SHARED_SERVICE_ERROR: Unknown sensitivity type for LEGACY RETRIEVE: %d\n", req->sensitivity);
        resp->status = -EINVAL;
        resp->data_size = 0;
    }
}


// --- Handler for VECTOR_ADD_REQUEST ---
static void handle_vector_add_request(
    const vector_add_request_payload_t* req_payload,
    vector_add_response_payload_t* resp_payload) {

    printf("SHARED_SERVICE_LOG: Handling VECTOR_ADD_REQUEST for %u elements.\n", req_payload->array_len_elements);

    if (req_payload->array_len_elements == 0 || req_payload->array_len_elements > VECTOR_ARRAY_MAX_ELEMENTS) {
        fprintf(stderr, "SHARED_SERVICE_ERROR: Invalid array_len_elements: %u\n", req_payload->array_len_elements);
        resp_payload->status = -EINVAL;
        resp_payload->array_len_elements = 0;
        return;
    }

    size_t array_data_size = req_payload->array_len_elements * sizeof(float);
    float* b_plain = NULL;
    float* c_plain = NULL;
    float* a_plain = NULL; // Result from CUDA
    int ret;

    // Allocate memory for plaintext arrays
    b_plain = (float*)malloc(array_data_size);
    c_plain = (float*)malloc(array_data_size);
    a_plain = (float*)malloc(array_data_size);

    if (!b_plain || !c_plain || !a_plain) {
        fprintf(stderr, "SHARED_SERVICE_ERROR: Failed to allocate memory for plaintext arrays.\n");
        resp_payload->status = -ENOMEM;
        goto cleanup;
    }

    // Decrypt B
    printf("  Decrypting array B...\n");
    ret = libos_aes_gcm_decrypt(g_shared_enclave_aes_key, req_payload->iv_b,
                                req_payload->masked_data_b, array_data_size, req_payload->tag_b,
                                (unsigned char*)b_plain, NULL, 0);
    if (ret != 0) {
        fprintf(stderr, "SHARED_SERVICE_ERROR: Decryption of array B failed: %d\n", ret);
        resp_payload->status = ret; // Propagate mbedTLS error
        goto cleanup;
    }

    // Decrypt C
    printf("  Decrypting array C...\n");
    ret = libos_aes_gcm_decrypt(g_shared_enclave_aes_key, req_payload->iv_c,
                                req_payload->masked_data_c, array_data_size, req_payload->tag_c,
                                (unsigned char*)c_plain, NULL, 0);
    if (ret != 0) {
        fprintf(stderr, "SHARED_SERVICE_ERROR: Decryption of array C failed: %d\n", ret);
        resp_payload->status = ret; // Propagate mbedTLS error
        goto cleanup;
    }
    printf("  Decryption successful.\n");

    // Perform CUDA operation
    printf("  Launching CUDA vector add kernel...\n");
    int cuda_err_code = 0;
    const char* cuda_err_str = NULL;
    ret = launch_vector_add_cuda(a_plain, b_plain, c_plain, req_payload->array_len_elements,
                                 &cuda_err_code, &cuda_err_str);
    if (ret != 0) {
        fprintf(stderr, "SHARED_SERVICE_ERROR: launch_vector_add_cuda failed. CUDA Error Code: %d, CUDA Error String: %s\n",
                cuda_err_code, cuda_err_str ? cuda_err_str : "Unknown CUDA error");
        resp_payload->status = cuda_err_code != 0 ? cuda_err_code : -1; // Use CUDA error if available, else generic
        goto cleanup;
    }
    printf("  CUDA vector add kernel successful.\n");

    // Encrypt A (result)
    printf("  Encrypting result array A...\n");
    generate_iv_for_response(resp_payload->iv_a, GCM_IV_SIZE_BYTES); // TODO: Use proper RNG
    ret = libos_aes_gcm_encrypt(g_shared_enclave_aes_key, resp_payload->iv_a,
                                (const unsigned char*)a_plain, array_data_size,
                                resp_payload->masked_data_a, resp_payload->tag_a,
                                NULL, 0);
    if (ret != 0) {
        fprintf(stderr, "SHARED_SERVICE_ERROR: Encryption of result array A failed: %d\n", ret);
        resp_payload->status = ret; // Propagate mbedTLS error
        goto cleanup;
    }
    printf("  Encryption of result array A successful.\n");

    // Prepare response
    resp_payload->status = 0; // Success
    resp_payload->array_len_elements = req_payload->array_len_elements;

cleanup:
    free(b_plain);
    free(c_plain);
    free(a_plain);
    printf("SHARED_SERVICE_LOG: Finished VECTOR_ADD_REQUEST handling with status %d.\n", resp_payload->status);
}


// --- IPC Server Functions using PAL ---

// Global handle for the listening pipe. Initialized in main.
static PAL_HANDLE g_listening_pipe_handle = PAL_HANDLE_INITIALIZER;


// Accepts a new client connection on the global listening pipe.
// Outputs the client's connection handle and its VMID.
int ipc_accept_client_connection(PAL_HANDLE* client_conn_handle, IDTYPE* client_vmid) {
    if (!g_listening_pipe_handle) {
        fprintf(stderr, "SHARED_SERVICE_ERROR: Server listening pipe not initialized.\n");
        return -EINVAL;
    }

    printf("SHARED_SERVICE_LOG: Waiting for a client connection on listening handle %p...\n", g_listening_pipe_handle);
    
    int ret = PalStreamAccept(g_listening_pipe_handle, client_conn_handle, /*timeout_ms=*/NULL);
    if (ret < 0) {
        fprintf(stderr, "SHARED_SERVICE_ERROR: PalStreamAccept failed: %s\n", pal_strerror(ret));
        return pal_to_unix_errno(ret);
    }
    printf("SHARED_SERVICE_LOG: Accepted a raw connection, client handle %p. Reading client VMID...\n", *client_conn_handle);

    // After accepting, the client is expected to send its VMID as the first message.
    // This mimics the client-side `ipc_connect` which sends `g_process_ipc_ids.self_vmid`.
    ret = read_exact(*client_conn_handle, client_vmid, sizeof(IDTYPE));
    if (ret < 0) {
        fprintf(stderr, "SHARED_SERVICE_ERROR: Failed to read client VMID from handle %p: %s\n", *client_conn_handle, unix_strerror(ret));
        PalObjectDestroy(client_conn_handle); // Clean up the accepted handle
        return ret;
    }

    printf("SHARED_SERVICE_LOG: Successfully accepted connection from client VMID %u (handle %p)\n", *client_vmid, *client_conn_handle);
    return 0; // Success
}

// Receives a raw message from a specific client connection.
// The caller is responsible for interpreting the message based on an expected structure.
// This is a simplified version. A robust version would read a header first to get payload size and type.
int ipc_receive_raw_message_from_client(PAL_HANDLE client_conn_handle, void* buffer, size_t buffer_len, size_t* bytes_received) {
    if (!client_conn_handle || !buffer || !bytes_received) return -EINVAL;

    printf("SHARED_SERVICE_LOG: Attempting to receive raw message on handle %p (buffer_len %zu)...\n", client_conn_handle, buffer_len);

    int ret = read_exact(client_conn_handle, buffer, buffer_len); // Attempt to read exactly buffer_len
    if (ret < 0) {
        if (ret == -ECONNRESET || ret == -EPIPE) {
            printf("SHARED_SERVICE_LOG: Client disconnected while trying to read raw (handle %p): %s\n", client_conn_handle, unix_strerror(ret));
        } else {
            fprintf(stderr, "SHARED_SERVICE_ERROR: raw read_exact failed on handle %p: %s\n", client_conn_handle, unix_strerror(ret));
        }
        *bytes_received = 0;
        return ret;
    }
    
    *bytes_received = buffer_len; 
    printf("SHARED_SERVICE_LOG: Successfully received %zu raw bytes on handle %p.\n", *bytes_received, client_conn_handle);
    return 0; // Success
}

// Sends a raw message back to a specific client.
int ipc_send_raw_response_to_client(PAL_HANDLE client_conn_handle, const void* buffer, size_t buffer_len) {
    if (!client_conn_handle || !buffer) return -EINVAL;
    if (buffer_len == 0) return 0; // Nothing to send

    printf("SHARED_SERVICE_LOG: Attempting to send raw response on handle %p (buffer_len %zu)...\n", client_conn_handle, buffer_len);

    int ret = write_exact(client_conn_handle, buffer, buffer_len);
    if (ret < 0) {
        fprintf(stderr, "SHARED_SERVICE_ERROR: raw write_exact failed on handle %p: %s\n", client_conn_handle, unix_strerror(ret));
        return ret;
    }
    printf("SHARED_SERVICE_LOG: Successfully sent raw response on handle %p.\n", client_conn_handle);
    return 0; // Success
}


void handle_client_session(PAL_HANDLE client_handle, IDTYPE client_vmid) {
    printf("SHARED_SERVICE_LOG: New session started for client VMID %u (handle %p)\n", client_vmid, client_handle);
    
    // For this example, we use a large enough buffer for libos_ipc_msg
    // and assume its 'data' field can hold our largest payload.
    // A more robust approach would dynamically allocate/check sizes.
    char ipc_buffer[sizeof(libos_ipc_msg_header_t) + sizeof(vector_add_request_payload_t)]; // Max expected request
    char ipc_response_buffer[sizeof(libos_ipc_msg_header_t) + sizeof(vector_add_response_payload_t)]; // Max expected response

    libos_ipc_msg_t* request_msg = (libos_ipc_msg_t*)ipc_buffer;
    libos_ipc_msg_t* response_msg = (libos_ipc_msg_t*)ipc_response_buffer;
    
    int ret;
    size_t received_size;

    while (1) { // Loop to handle multiple requests per client session
        memset(ipc_buffer, 0, sizeof(ipc_buffer));
        memset(ipc_response_buffer, 0, sizeof(ipc_response_buffer));

        // 1. Read the libos_ipc_msg_header_t first to determine payload size and type
        ret = ipc_receive_raw_message_from_client(client_handle, request_msg, sizeof(libos_ipc_msg_header_t), &received_size);
        if (ret < 0 || received_size != sizeof(libos_ipc_msg_header_t)) {
            fprintf(stderr, "SHARED_SERVICE_INFO: Failed to read IPC message header or client disconnected. Ending session for client %u.\n", client_vmid);
            break;
        }

        operation_type_t op_type = (operation_type_t)GET_UNALIGNED(request_msg->header.code);
        size_t total_expected_size = GET_UNALIGNED(request_msg->header.size);
        size_t payload_expected_size = total_expected_size - sizeof(libos_ipc_msg_header_t);

        printf("SHARED_SERVICE_LOG: Client %u: Received header. OpType: %d, TotalSize: %zu, PayloadSize: %zu\n",
               client_vmid, op_type, total_expected_size, payload_expected_size);

        // 2. Read the actual payload based on header information
        if (payload_expected_size > 0) {
            if (payload_expected_size > sizeof(request_msg->data)) { // Check if our buffer can hold it
                 fprintf(stderr, "SHARED_SERVICE_ERROR: Client %u: Payload size %zu too large for buffer %zu. Ending session.\n",
                        client_vmid, payload_expected_size, sizeof(request_msg->data));
                break;
            }
            ret = ipc_receive_raw_message_from_client(client_handle, request_msg->data, payload_expected_size, &received_size);
            if (ret < 0 || received_size != payload_expected_size) {
                fprintf(stderr, "SHARED_SERVICE_INFO: Failed to read IPC message payload or client disconnected. Ending session for client %u.\n", client_vmid);
                break;
            }
        }
        
        // 3. Dispatch based on operation type
        if (op_type == VECTOR_ADD_REQUEST) {
            if (payload_expected_size != sizeof(vector_add_request_payload_t)) {
                fprintf(stderr, "SHARED_SERVICE_ERROR: Client %u: VECTOR_ADD_REQUEST payload size mismatch. Expected %zu, Got %zu.\n",
                        client_vmid, sizeof(vector_add_request_payload_t), payload_expected_size);
                // Send generic error back? For now, just break.
                break;
            }
            vector_add_request_payload_t* req_payload = (vector_add_request_payload_t*)request_msg->data;
            vector_add_response_payload_t resp_payload; // Stack allocate response payload
            memset(&resp_payload, 0, sizeof(resp_payload));

            handle_vector_add_request(req_payload, &resp_payload);

            // Prepare response IPC message
            init_ipc_msg(response_msg, VECTOR_ADD_REQUEST, // Or a new VECTOR_ADD_RESPONSE code
                         sizeof(libos_ipc_msg_header_t) + sizeof(vector_add_response_payload_t));
            memcpy(response_msg->data, &resp_payload, sizeof(vector_add_response_payload_t));
            
            ret = ipc_send_raw_response_to_client(client_handle, response_msg, GET_UNALIGNED(response_msg->header.size));

        } else if (op_type == STORE_DATA || op_type == RETRIEVE_DATA) {
            // Example for legacy operations (assuming they fit in legacy_data_request_t payload size)
            if (payload_expected_size > sizeof(legacy_data_request_t)) { // Check specific size for legacy
                 fprintf(stderr, "SHARED_SERVICE_ERROR: Client %u: Legacy op payload size %zu too large.\n",
                        client_vmid, payload_expected_size);
                break;
            }
            legacy_data_request_t* legacy_req = (legacy_data_request_t*)request_msg->data;
            legacy_data_response_t legacy_resp;
            memset(&legacy_resp, 0, sizeof(legacy_resp));

            if (op_type == STORE_DATA) {
                handle_legacy_store_request(legacy_req, &legacy_resp);
            } else { // RETRIEVE_DATA
                handle_legacy_retrieve_request(legacy_req, &legacy_resp);
            }
            
            init_ipc_msg(response_msg, op_type, // Echo back op_type or specific response code
                         sizeof(libos_ipc_msg_header_t) + sizeof(legacy_data_response_t));
            memcpy(response_msg->data, &legacy_resp, sizeof(legacy_data_response_t));
            ret = ipc_send_raw_response_to_client(client_handle, response_msg, GET_UNALIGNED(response_msg->header.size));

        } else {
            fprintf(stderr, "SHARED_SERVICE_ERROR: Client %u: Unknown operation type: %d\n", client_vmid, op_type);
            // Send generic error back?
            // For now, just break.
            ret = -1; // Mark error to break loop
        }
        
        if (ret < 0) {
            fprintf(stderr, "SHARED_SERVICE_INFO: Ending session for client %u due to send error or unhandled op.\n", client_vmid);
            break; 
        }
    }
    
    PalObjectDestroy(&client_handle); // Close client-specific handle
    printf("SHARED_SERVICE_LOG: Session ended for client VMID %u, handle %p destroyed.\n", client_vmid, client_handle);
}


int main(int argc, char *argv[]) {
    printf("SHARED_SERVICE_LOG: Starting Data Storage Service...\n");

    if (init_ipc() < 0) { // Initializes g_process_ipc_ids
        fprintf(stderr, "SHARED_SERVICE_FATAL: Failed to initialize IPC system (init_ipc).\n");
        return EXIT_FAILURE;
    }
    printf("SHARED_SERVICE_LOG: IPC system initialized via init_ipc() (self vmid: %u, parent vmid: %u).\n",
           g_process_ipc_ids.self_vmid, g_process_ipc_ids.parent_vmid);

    // Construct the listening URI for this service
    char listening_uri[PIPE_URI_SIZE];
    int snprintf_ret = snprintf(listening_uri, sizeof(listening_uri), URI_PREFIX_PIPE "%lu/%u",
                                g_pal_public_state->instance_id, g_process_ipc_ids.self_vmid);
    if (snprintf_ret < 0 || (size_t)snprintf_ret >= sizeof(listening_uri)) {
        fprintf(stderr, "SHARED_SERVICE_FATAL: Failed to construct listening URI.\n");
        return EXIT_FAILURE;
    }
    printf("SHARED_SERVICE_LOG: Service will listen on URI: %s\n", listening_uri);

    // Create and start listening on the PAL pipe
    // PAL_LISTEN_DEFAULT should allow multiple connections to be queued.
    int pal_ret = PalStreamListen(listening_uri, PAL_LISTEN_DEFAULT, &g_listening_pipe_handle);
    if (pal_ret < 0) {
        fprintf(stderr, "SHARED_SERVICE_FATAL: PalStreamListen failed for URI %s: %s\n",
                listening_uri, pal_strerror(pal_ret));
        return EXIT_FAILURE;
    }
    printf("SHARED_SERVICE_LOG: Successfully listening on handle %p for URI %s\n", g_listening_pipe_handle, listening_uri);

    if (mkdir(UNTRUSTED_SHM_PATH_PREFIX, 0775) != 0 && errno != EEXIST) {
        perror("SHARED_SERVICE_FATAL: mkdir failed for UNTRUSTED_SHM_PATH_PREFIX");
        PalObjectDestroy(&g_listening_pipe_handle); // Clean up listening handle
        return EXIT_FAILURE;
    }
    printf("SHARED_SERVICE_LOG: Untrusted SHM path prefix '%s' ensured.\n", UNTRUSTED_SHM_PATH_PREFIX);

    printf("SHARED_SERVICE_LOG: Entering main client acceptance loop...\n");
    while (1) {
        PAL_HANDLE client_conn_handle = PAL_HANDLE_INITIALIZER; // Important to initialize
        IDTYPE client_vmid = 0;

        int accept_ret = ipc_accept_client_connection(&client_conn_handle, &client_vmid);
        
        if (accept_ret < 0) {
            // Error already logged by ipc_accept_client_connection
            fprintf(stderr, "SHARED_SERVICE_WARNING: Failed to accept a client connection (%s). Continuing to listen...\n",
                    unix_strerror(accept_ret));
            // Optionally, add a small delay here if accept failures are rapid and persistent
            // sleep(1); 
            continue;
        }

        // At this point, client_conn_handle is valid and client_vmid is known.
        // Handle all communications with this client serially.
        // For concurrent clients, a new thread or asynchronous handling would be needed here.
        handle_client_session(client_conn_handle, client_vmid);
        // client_conn_handle is destroyed at the end of handle_client_session
    }

    printf("SHARED_SERVICE_LOG: Data Storage Service shutting down (this part of code is unreachable in normal server loop).\n");
    if (g_listening_pipe_handle) { // Check if it was initialized
        PalObjectDestroy(&g_listening_pipe_handle); // Clean up listening handle
    }
    return EXIT_SUCCESS;
}

// --- Service Logic Unit Tests ---

static int test_shm_store_retrieve() {
    printf("\nRunning test_shm_store_retrieve...\n");
    data_request_t store_req;
    data_response_t store_resp;
    data_request_t retrieve_req;
    data_response_t retrieve_resp;

    const char* test_filename = "unit_test_shm_file.txt";
    const char* test_content = "Hello SHM from unit test!";
    size_t test_content_len = strlen(test_content);

    // Prepare store request
    memset(&store_req, 0, sizeof(store_req));
    store_req.operation = STORE_DATA;
    store_req.sensitivity = SENSITIVITY_LOW_SHM;
    strncpy(store_req.path, test_filename, MAX_PATH_SIZE -1);
    store_req.data_size = test_content_len;
    memcpy(store_req.data, test_content, test_content_len);

    // Handle store request
    printf("  Calling handle_store_request for SHM...\n");
    handle_store_request(&store_req, &store_resp);
    assert(store_resp.status == 0 && "SHM store request failed");
    if (store_resp.status != 0) {
        fprintf(stderr, "  SHM store failed with status: %d\n", store_resp.status);
        return -1;
    }

    // Verify file creation and content
    char full_shm_path[MAX_PATH_SIZE + sizeof(UNTRUSTED_SHM_PATH_PREFIX) + 1];
    snprintf(full_shm_path, sizeof(full_shm_path), "%s/%s", UNTRUSTED_SHM_PATH_PREFIX, test_filename);
    printf("  Verifying SHM file: %s\n", full_shm_path);

    FILE* fp = fopen(full_shm_path, "rb");
    assert(fp != NULL && "Failed to open SHM file for verification");
    if (!fp) {
        perror("  fopen for SHM verification failed");
        return -1;
    }
    char read_buffer[100];
    memset(read_buffer, 0, sizeof(read_buffer));
    size_t bytes_read = fread(read_buffer, 1, sizeof(read_buffer) -1, fp);
    fclose(fp);

    assert(bytes_read == test_content_len && "SHM file content length mismatch");
    assert(memcmp(test_content, read_buffer, test_content_len) == 0 && "SHM file content mismatch");
    if (bytes_read != test_content_len || memcmp(test_content, read_buffer, test_content_len) != 0) {
        fprintf(stderr, "  SHM file verification failed. Expected: '%s', Got: '%s'\n", test_content, read_buffer);
        return -1;
    }
    printf("  SHM file content verified.\n");

    // Prepare retrieve request
    memset(&retrieve_req, 0, sizeof(retrieve_req));
    retrieve_req.operation = RETRIEVE_DATA;
    retrieve_req.sensitivity = SENSITIVITY_LOW_SHM;
    strncpy(retrieve_req.path, test_filename, MAX_PATH_SIZE -1);

    // Handle retrieve request
    printf("  Calling handle_retrieve_request for SHM...\n");
    handle_retrieve_request(&retrieve_req, &retrieve_resp);
    assert(retrieve_resp.status == 0 && "SHM retrieve request failed");
    if (retrieve_resp.status != 0) {
        fprintf(stderr, "  SHM retrieve failed with status: %d\n", retrieve_resp.status);
        return -1; // Don't proceed to cleanup if retrieve failed, to inspect the file
    }
    assert(retrieve_resp.data_size == test_content_len && "Retrieved SHM data length mismatch");
    assert(memcmp(test_content, retrieve_resp.data, test_content_len) == 0 && "Retrieved SHM data content mismatch");
     if (retrieve_resp.data_size != test_content_len || memcmp(test_content, retrieve_resp.data, test_content_len) != 0) {
        fprintf(stderr, "  SHM retrieve content verification failed.\n");
        return -1;
    }
    printf("  SHM retrieve content verified.\n");

    // Cleanup
    printf("  Cleaning up SHM file: %s\n", full_shm_path);
    if (remove(full_shm_path) != 0) {
        perror("  Failed to remove SHM test file");
        // Not failing the test for cleanup failure, but logging it.
    }

    printf("test_shm_store_retrieve: PASSED\n");
    return 0;
}

static int test_gpu_store_placeholder() {
    printf("\nRunning test_gpu_store_placeholder...\n");
    data_request_t req;
    data_response_t resp;

    memset(&req, 0, sizeof(req));
    req.operation = STORE_DATA;
    req.sensitivity = SENSITIVITY_MEDIUM_GPU;
    strncpy(req.path, "gpu_test_id_1", MAX_PATH_SIZE -1);
    req.data_size = 10;
    memset(req.data, 'G', 10);

    printf("  Calling handle_store_request for GPU (placeholder)...\n");
    handle_store_request(&req, &resp);
    // Placeholder store_to_gpu returns 0 for success
    assert(resp.status == 0 && "GPU store placeholder request failed");
    if (resp.status != 0) {
         fprintf(stderr, "  GPU store placeholder failed with status: %d\n", resp.status);
        return -1;
    }

    printf("test_gpu_store_placeholder: PASSED (relies on placeholder returning success)\n");
    return 0;
}

static int test_gpu_retrieve_placeholder() {
    printf("\nRunning test_gpu_retrieve_placeholder...\n");
    data_request_t req;
    data_response_t resp;

    memset(&req, 0, sizeof(req));
    req.operation = RETRIEVE_DATA;
    req.sensitivity = SENSITIVITY_MEDIUM_GPU;
    strncpy(req.path, "gpu_test_id_2", MAX_PATH_SIZE-1);

    printf("  Calling handle_retrieve_request for GPU (placeholder)...\n");
    handle_retrieve_request(&req, &resp);
    // Placeholder retrieve_from_gpu returns 0 for success and dummy data
    assert(resp.status == 0 && "GPU retrieve placeholder request failed");
    if (resp.status != 0) {
         fprintf(stderr, "  GPU retrieve placeholder failed with status: %d\n", resp.status);
        return -1;
    }
    // Could also check resp.data_size and resp.data for dummy values if they are consistent
    // For now, just checking status is enough for placeholder.
    printf("  GPU retrieve placeholder returned data size: %u\n", resp.data_size);


    printf("test_gpu_retrieve_placeholder: PASSED (relies on placeholder returning success)\n");
    return 0;
}

static int run_service_logic_tests() {
    printf("\n--- Running Shared Service Logic Unit Tests ---\n");
    int overall_status = 0;

    if (test_shm_store_retrieve() != 0) {
        overall_status = -1;
    }
    if (test_gpu_store_placeholder() != 0) {
        overall_status = -1;
    }
    if (test_gpu_retrieve_placeholder() != 0) {
        overall_status = -1;
    }

    if (overall_status == 0) {
        printf("\nAll shared_service logic tests PASSED.\n");
    } else {
        printf("\nOne or more shared_service logic tests FAILED.\n");
    }
    printf("--- Finished Shared Service Logic Unit Tests ---\n");
    return overall_status;
}


int main(int argc, char *argv[]) {
    if (argc > 1 && strcmp(argv[1], "--run-tests") == 0) {
        // Ensure SHM directory exists for tests, as main server logic might not run.
        if (mkdir(UNTRUSTED_SHM_PATH_PREFIX, 0775) != 0 && errno != EEXIST) {
            perror("SHARED_SERVICE_TEST_FATAL: mkdir failed for UNTRUSTED_SHM_PATH_PREFIX before tests");
            return EXIT_FAILURE;
        }
        return run_service_logic_tests();
    }

    printf("SHARED_SERVICE_LOG: Starting Data Storage Service...\n");

    if (init_ipc() < 0) { // Initializes g_process_ipc_ids
        fprintf(stderr, "SHARED_SERVICE_FATAL: Failed to initialize IPC system (init_ipc).\n");
        return EXIT_FAILURE;
    }
    printf("SHARED_SERVICE_LOG: IPC system initialized via init_ipc() (self vmid: %u, parent vmid: %u).\n",
           g_process_ipc_ids.self_vmid, g_process_ipc_ids.parent_vmid);

    // Construct the listening URI for this service
    char listening_uri[PIPE_URI_SIZE];
    int snprintf_ret = snprintf(listening_uri, sizeof(listening_uri), URI_PREFIX_PIPE "%lu/%u",
                                g_pal_public_state->instance_id, g_process_ipc_ids.self_vmid);
    if (snprintf_ret < 0 || (size_t)snprintf_ret >= sizeof(listening_uri)) {
        fprintf(stderr, "SHARED_SERVICE_FATAL: Failed to construct listening URI.\n");
        return EXIT_FAILURE;
    }
    printf("SHARED_SERVICE_LOG: Service will listen on URI: %s\n", listening_uri);

    // Create and start listening on the PAL pipe
    // PAL_LISTEN_DEFAULT should allow multiple connections to be queued.
    int pal_ret = PalStreamListen(listening_uri, PAL_LISTEN_DEFAULT, &g_listening_pipe_handle);
    if (pal_ret < 0) {
        fprintf(stderr, "SHARED_SERVICE_FATAL: PalStreamListen failed for URI %s: %s\n",
                listening_uri, pal_strerror(pal_ret));
        return EXIT_FAILURE;
    }
    printf("SHARED_SERVICE_LOG: Successfully listening on handle %p for URI %s\n", g_listening_pipe_handle, listening_uri);

    if (mkdir(UNTRUSTED_SHM_PATH_PREFIX, 0775) != 0 && errno != EEXIST) {
        perror("SHARED_SERVICE_FATAL: mkdir failed for UNTRUSTED_SHM_PATH_PREFIX");
        PalObjectDestroy(&g_listening_pipe_handle); // Clean up listening handle
        return EXIT_FAILURE;
    }
    printf("SHARED_SERVICE_LOG: Untrusted SHM path prefix '%s' ensured.\n", UNTRUSTED_SHM_PATH_PREFIX);

    printf("SHARED_SERVICE_LOG: Entering main client acceptance loop...\n");
    while (1) {
        PAL_HANDLE client_conn_handle = PAL_HANDLE_INITIALIZER; // Important to initialize
        IDTYPE client_vmid = 0;

        int accept_ret = ipc_accept_client_connection(&client_conn_handle, &client_vmid);
        
        if (accept_ret < 0) {
            // Error already logged by ipc_accept_client_connection
            fprintf(stderr, "SHARED_SERVICE_WARNING: Failed to accept a client connection (%s). Continuing to listen...\n",
                    unix_strerror(accept_ret));
            // Optionally, add a small delay here if accept failures are rapid and persistent
            // sleep(1); 
            continue;
        }

        // At this point, client_conn_handle is valid and client_vmid is known.
        // Handle all communications with this client serially.
        // For concurrent clients, a new thread or asynchronous handling would be needed here.
        handle_client_session(client_conn_handle, client_vmid);
        // client_conn_handle is destroyed at the end of handle_client_session
    }

    printf("SHARED_SERVICE_LOG: Data Storage Service shutting down (this part of code is unreachable in current loop).\n");
    PalObjectDestroy(&g_listening_pipe_handle); // Clean up listening handle
    return EXIT_SUCCESS;
}
