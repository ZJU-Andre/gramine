#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h> // For sleep, close
#include <sys/stat.h> // For mkdir

// Assuming these are the locations of necessary Gramine headers
#include "libos_ipc.h"   // For IPC functions (conceptualized based on previous work)
#include "libos_types.h" // For IDTYPE and other types
#include "pal.h"         // For PAL_HANDLE, PalGetSecurityValidatedTime etc.


#define MAX_DATA_SIZE 1024
#define MAX_PATH_SIZE 256
#define UNTRUSTED_SHM_PATH_PREFIX "/untrusted_region"

typedef enum {
    STORE_DATA,
    RETRIEVE_DATA
} operation_type_t;

typedef enum {
    SENSITIVITY_MEDIUM_GPU, // Data that might be processed/stored via GPU paths
    SENSITIVITY_LOW_SHM     // Data suitable for storage in untrusted shared memory
} data_sensitivity_t;

typedef struct {
    operation_type_t operation;
    data_sensitivity_t sensitivity;
    char path[MAX_PATH_SIZE]; // Identifier for GPU data or relative path for SHM
    uint32_t data_size;
    unsigned char data[MAX_DATA_SIZE]; // Payload
} data_request_t;

typedef struct {
    int status; // 0 for success, negative errno for errors
    uint32_t data_size;
    unsigned char data[MAX_DATA_SIZE]; // For retrieved data
} data_response_t;

// --- Placeholder GPU Functions ---
int store_to_gpu(const char* id, const unsigned char* data, uint32_t size) {
    printf("SHARED_SERVICE_LOG: Storing to GPU (ID: %s, Size: %u) - Placeholder\n", id, size);
    // In a real scenario, this would involve GPU driver calls, CUDA/OpenCL operations, etc.
    if (!id || !data) return -EINVAL;
    if (size > MAX_DATA_SIZE) return -EFBIG; // Example error
    return 0; // Success
}

int retrieve_from_gpu(const char* id, unsigned char* buffer, uint32_t* size_read) {
    printf("SHARED_SERVICE_LOG: Retrieving from GPU (ID: %s) - Placeholder\n", id);
    // In a real scenario, this would involve GPU driver calls to fetch data.
    if (!id || !buffer || !size_read) return -EINVAL;
    
    // Simulate finding some data
    const char* dummy_data = "SampleGPUData";
    uint32_t dummy_size = strlen(dummy_data);
    if (dummy_size > MAX_DATA_SIZE) dummy_size = MAX_DATA_SIZE; // Ensure fit

    memcpy(buffer, dummy_data, dummy_size);
    *size_read = dummy_size;
    
    printf("SHARED_SERVICE_LOG: Retrieved %u bytes from GPU (ID: %s)\n", *size_read, id);
    return 0; // Success
}

// --- Request Handlers ---
void handle_store_request(const data_request_t* req, data_response_t* resp) {
    resp->data_size = 0; // Typically no data in response for store, unless an ID is returned

    if (req->sensitivity == SENSITIVITY_MEDIUM_GPU) {
        printf("SHARED_SERVICE_LOG: Handling STORE request for GPU (Path/ID: %s, Size: %u)\n", req->path, req->data_size);
        resp->status = store_to_gpu(req->path, req->data, req->data_size);
    } else if (req->sensitivity == SENSITIVITY_LOW_SHM) {
        char full_path[MAX_PATH_SIZE + sizeof(UNTRUSTED_SHM_PATH_PREFIX) + 1];
        snprintf(full_path, sizeof(full_path), "%s/%s", UNTRUSTED_SHM_PATH_PREFIX, req->path);
        printf("SHARED_SERVICE_LOG: Handling STORE request for SHM (Full Path: %s, Size: %u)\n", full_path, req->data_size);

        // Ensure directory exists (basic version, only one level deep for simplicity)
        // In a real service, you might need a more robust way to handle paths/directories.
        char* last_slash = strrchr(full_path, '/');
        if (last_slash) {
            *last_slash = '\0'; // Temporarily cut string to get dir path
            // Use S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH for 0775
            if (mkdir(full_path, 0775) != 0 && errno != EEXIST) {
                perror("SHARED_SERVICE_ERROR: mkdir failed for SHM path");
                resp->status = -errno;
                *last_slash = '/'; // Restore path
                return;
            }
            *last_slash = '/'; // Restore path
        }
        
        FILE* fp = fopen(full_path, "wb");
        if (!fp) {
            perror("SHARED_SERVICE_ERROR: fopen for SHM store failed");
            resp->status = -errno;
            return;
        }
        size_t written = fwrite(req->data, 1, req->data_size, fp);
        if (written != req->data_size) {
            perror("SHARED_SERVICE_ERROR: fwrite for SHM store failed");
            resp->status = ferror(fp) ? -EIO : -ENOSPC; // Example errors
            fclose(fp);
            return;
        }
        fclose(fp);
        printf("SHARED_SERVICE_LOG: Successfully stored %zu bytes to SHM path %s\n", written, full_path);
        resp->status = 0;
    } else {
        fprintf(stderr, "SHARED_SERVICE_ERROR: Unknown sensitivity type for STORE: %d\n", req->sensitivity);
        resp->status = -EINVAL;
    }
}

void handle_retrieve_request(const data_request_t* req, data_response_t* resp) {
    if (req->sensitivity == SENSITIVITY_MEDIUM_GPU) {
        printf("SHARED_SERVICE_LOG: Handling RETRIEVE request for GPU (Path/ID: %s)\n", req->path);
        resp->status = retrieve_from_gpu(req->path, resp->data, &resp->data_size);
    } else if (req->sensitivity == SENSITIVITY_LOW_SHM) {
        char full_path[MAX_PATH_SIZE + sizeof(UNTRUSTED_SHM_PATH_PREFIX) + 1];
        snprintf(full_path, sizeof(full_path), "%s/%s", UNTRUSTED_SHM_PATH_PREFIX, req->path);
        printf("SHARED_SERVICE_LOG: Handling RETRIEVE request for SHM (Full Path: %s)\n", full_path);

        FILE* fp = fopen(full_path, "rb");
        if (!fp) {
            perror("SHARED_SERVICE_ERROR: fopen for SHM retrieve failed");
            resp->status = -errno;
            resp->data_size = 0;
            return;
        }
        size_t read_bytes = fread(resp->data, 1, MAX_DATA_SIZE, fp);
        if (ferror(fp)) {
            perror("SHARED_SERVICE_ERROR: fread for SHM retrieve failed");
            resp->status = -EIO;
            resp->data_size = 0;
            fclose(fp);
            return;
        }
        fclose(fp);
        resp->data_size = read_bytes;
        printf("SHARED_SERVICE_LOG: Successfully retrieved %u bytes from SHM path %s\n", resp->data_size, full_path);
        resp->status = 0;
    } else {
        fprintf(stderr, "SHARED_SERVICE_ERROR: Unknown sensitivity type for RETRIEVE: %d\n", req->sensitivity);
        resp->status = -EINVAL;
        resp->data_size = 0;
    }
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

// Receives a message (data_request_t) from a specific client connection.
// Uses the client_conn_handle obtained from ipc_accept_client_connection.
int ipc_receive_message_from_client(PAL_HANDLE client_conn_handle, data_request_t* request, size_t* bytes_received) {
    if (!client_conn_handle) return -EINVAL;

    printf("SHARED_SERVICE_LOG: Attempting to receive message on handle %p...\n", client_conn_handle);

    // The client sends a libos_ipc_msg structure. We need to read its header first.
    // However, the existing client-side `ipc_send_message` and `ipc_send_msg_and_get_response`
    // in `libos_ipc.c` directly write the `libos_ipc_msg` (which includes the payload).
    // For simplicity, let's assume the client just sends the raw `data_request_t` structure
    // if we are not using `libos_ipc_msg` for client-to-server raw data transfer.
    // This is a divergence from how `libos_ipc.c` is structured for request/response with sequence numbers.
    // For this service, we'll assume direct struct transfer for now.
    // A more robust implementation would use the `libos_ipc_msg` framing.

    int ret = read_exact(client_conn_handle, request, sizeof(data_request_t));
    if (ret < 0) {
        if (ret == -ECONNRESET || ret == -EPIPE) { // Common errors for client disconnection
            printf("SHARED_SERVICE_LOG: Client disconnected while trying to read (handle %p): %s\n", client_conn_handle, unix_strerror(ret));
        } else {
            fprintf(stderr, "SHARED_SERVICE_ERROR: read_exact failed on handle %p: %s\n", client_conn_handle, unix_strerror(ret));
        }
        *bytes_received = 0; // Indicate error or disconnection
        return ret;
    }
    
    // If read_exact returns 0, it means EOF (client closed connection gracefully before sending full message part)
    // However, read_exact is designed to return an error if not all bytes are read.
    // So, a success (ret == 0) from read_exact implies all bytes for sizeof(data_request_t) were read.
    *bytes_received = sizeof(data_request_t); 
    printf("SHARED_SERVICE_LOG: Successfully received %zu bytes on handle %p.\n", *bytes_received, client_conn_handle);
    return 0; // Success
}

// Sends a response (data_response_t) back to a specific client.
// Uses the client_conn_handle.
int ipc_send_response_to_client(PAL_HANDLE client_conn_handle, const data_response_t* response) {
    if (!client_conn_handle) return -EINVAL;

    printf("SHARED_SERVICE_LOG: Attempting to send response on handle %p (status %d, size %u)...\n",
           client_conn_handle, response->status, response->data_size);

    // Similar to receive, assuming direct struct transfer.
    int ret = write_exact(client_conn_handle, response, sizeof(data_response_t));
    if (ret < 0) {
        fprintf(stderr, "SHARED_SERVICE_ERROR: write_exact failed on handle %p: %s\n", client_conn_handle, unix_strerror(ret));
        return ret;
    }
    printf("SHARED_SERVICE_LOG: Successfully sent response on handle %p.\n", client_conn_handle);
    return 0; // Success
}

void handle_client_session(PAL_HANDLE client_handle, IDTYPE client_vmid) {
    printf("SHARED_SERVICE_LOG: New session started for client VMID %u (handle %p)\n", client_vmid, client_handle);
    data_request_t request;
    data_response_t response;
    int ret;
    size_t received_size;

    while (1) { // Loop to handle multiple requests per client session
        memset(&request, 0, sizeof(request));
        memset(&response, 0, sizeof(response));

        ret = ipc_receive_message_from_client(client_handle, &request, &received_size);
        
        if (ret < 0) {
            // Error already logged by ipc_receive_message_from_client
            // ret could be -ECONNRESET, -EPIPE for disconnection, or other read errors
            fprintf(stderr, "SHARED_SERVICE_INFO: Ending session for client %u due to receive error %s.\n", client_vmid, unix_strerror(ret));
            break; 
        }
        if (received_size == 0) { 
            // This condition might not be hit if read_exact always returns error on partial read/EOF
            printf("SHARED_SERVICE_LOG: Client %u disconnected gracefully (received 0 bytes). Ending session.\n", client_vmid);
            break;
        }
        // Basic validation of received size. 
        // If using fixed-size structs, it should match exactly.
        if (received_size != sizeof(data_request_t)) {
             fprintf(stderr, "SHARED_SERVICE_ERROR: Received message of unexpected size %zu from client %u (expected %zu). Ending session.\n",
                     received_size, client_vmid, sizeof(data_request_t));
             break;
        }

        printf("SHARED_SERVICE_LOG: Client %u (handle %p): Received Op=%d, Sens=%d, Path='%s', DataSize=%u\n",
               client_vmid, client_handle, request.operation, request.sensitivity, request.path, request.data_size);

        if (request.operation == STORE_DATA) {
            handle_store_request(&request, &response);
        } else if (request.operation == RETRIEVE_DATA) {
            handle_retrieve_request(&request, &response);
        } else {
            fprintf(stderr, "SHARED_SERVICE_ERROR: Client %u: Unknown operation type: %d\n", client_vmid, request.operation);
            response.status = -EINVAL; // Invalid argument (unknown operation)
            response.data_size = 0;
        }
        
        printf("SHARED_SERVICE_LOG: Client %u (handle %p): Sending Resp: Status=%d, DataSize=%u\n",
               client_vmid, client_handle, response.status, response.data_size);

        ret = ipc_send_response_to_client(client_handle, &response);
        if (ret < 0) {
            // Error already logged by ipc_send_response_to_client
            fprintf(stderr, "SHARED_SERVICE_INFO: Ending session for client %u due to send error %s.\n", client_vmid, unix_strerror(ret));
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
