#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h> // For read, write, close, unlink
#include <fcntl.h>  // For O_RDWR, O_CREAT, shm_open
#include <sys/mman.h> // For mmap, munmap, shm_open
#include <sys/stat.h> // For mode constants, mkfifo
#include <errno.h>    // For errno

#include <cuda.h>
#include <mbedtls/gcm.h>
#include <mbedtls/error.h> // For mbedtls_strerror

// Include the IPC structures defined in LibOS
#include "../libos/include/libos_gpu_ipc.h" // Adjusted path

// --- Global variables ---
static int g_pipe_enclave_to_proxy_fd = -1;
static int g_pipe_proxy_to_enclave_fd = -1;
static void* g_gpu_shared_mem = NULL;
static size_t g_gpu_shm_size = 0;
// SHM path on the host system, derived from manifest uri.
static char g_shm_host_uri_path[MAX_SHM_PATH_LEN] = "/tmp/gramine_gpu_shm_file"; // Default, should align with manifest

CUdevice cuDevice;
CUcontext cuContext;

static uint8_t g_proxy_session_key[32];
static bool g_proxy_key_initialized = false;
static mbedtls_gcm_context g_proxy_aes_gcm_ctx;
static uint64_t g_proxy_encryption_invocation_count = 0; // For IV generation

// Opaque handle mapping for CUdeviceptr
#define MAX_GPU_MAPPINGS 128
typedef struct {
    uint64_t opaque_handle;
    CUdeviceptr dptr;
    size_t size;
} gpu_mem_mapping_t;
static gpu_mem_mapping_t g_gpu_mem_mappings[MAX_GPU_MAPPINGS];
static int g_gpu_mapping_count = 0;
static uint64_t g_next_opaque_handle = 1;

// Fixed pipe names
#define PIPE_ENCLAVE_TO_PROXY "/tmp/gramine_gpu_ipc_e2p"
#define PIPE_PROXY_TO_ENCLAVE "/tmp/gramine_gpu_ipc_p2e"

// --- Function Prototypes ---
int init_cuda_and_ipc(void);
void cleanup_cuda_and_ipc(void);
static int send_response(int pipe_fd, gpu_message_header_t* header, void* payload);
static CUdeviceptr get_dptr_from_opaque(uint64_t opaque_handle);
static uint64_t add_mapping(CUdeviceptr dptr, size_t size);
static int remove_mapping(uint64_t opaque_handle);
static void proxy_construct_iv(uint8_t* iv_buf);

// Command Handlers
int handle_malloc_device_cmd(const cmd_malloc_device_req_t* req, cmd_malloc_device_resp_t* resp);
int handle_free_device_cmd(const cmd_free_device_req_t* req);
int handle_memcpy_h2d_cmd(const cmd_memcpy_h2d_req_t* req);
int handle_memcpy_d2h_cmd(const cmd_memcpy_d2h_req_t* req, cmd_memcpy_d2h_resp_t* resp_payload);
int handle_launch_kernel_cmd(const cmd_launch_kernel_req_t* req);

// --- Helper Functions ---

static void mbedtls_print_error(const char* func_name, int mbedtls_err) {
    char err_buf[128];
    mbedtls_strerror(mbedtls_err, err_buf, sizeof(err_buf));
    fprintf(stderr, "PROXY ERROR: %s failed: %s (0x%04X)\n", func_name, err_buf, -mbedtls_err);
}

static int send_response(int pipe_fd, gpu_message_header_t* header, void* payload) {
    ssize_t bytes_written;
    bytes_written = write(pipe_fd, header, sizeof(gpu_message_header_t));
    if (bytes_written != sizeof(gpu_message_header_t)) {
        perror("PROXY ERROR: send_response - write header failed");
        return -1;
    }
    if (header->payload_size > 0 && payload) {
        bytes_written = write(pipe_fd, payload, header->payload_size);
        if (bytes_written != (ssize_t)header->payload_size) {
            perror("PROXY ERROR: send_response - write payload failed");
            return -1;
        }
    }
    return 0;
}

static void proxy_construct_iv(uint8_t* iv_buf) {
    // Simple counter-based IV. First 8 bytes are the counter, rest zeros.
    // Ensure g_proxy_encryption_invocation_count is incremented before each use for encryption.
    memset(iv_buf, 0, 12);
    uint64_t counter = g_proxy_encryption_invocation_count;
    memcpy(iv_buf, &counter, sizeof(uint64_t));
}

static CUdeviceptr get_dptr_from_opaque(uint64_t opaque_handle) {
    for (int i = 0; i < g_gpu_mapping_count; ++i) {
        if (g_gpu_mem_mappings[i].opaque_handle == opaque_handle) {
            return g_gpu_mem_mappings[i].dptr;
        }
    }
    fprintf(stderr, "PROXY ERROR: No mapping found for opaque_handle %lu\n", opaque_handle);
    return (CUdeviceptr)0;
}

static uint64_t add_mapping(CUdeviceptr dptr, size_t size) {
    if (g_gpu_mapping_count >= MAX_GPU_MAPPINGS) {
        fprintf(stderr, "PROXY ERROR: Max GPU mappings reached (%d)\n", MAX_GPU_MAPPINGS);
        return 0; // Invalid handle
    }
    uint64_t current_handle = g_next_opaque_handle++;
    g_gpu_mem_mappings[g_gpu_mapping_count].opaque_handle = current_handle;
    g_gpu_mem_mappings[g_gpu_mapping_count].dptr = dptr;
    g_gpu_mem_mappings[g_gpu_mapping_count].size = size;
    g_gpu_mapping_count++;
    return current_handle;
}

static int remove_mapping(uint64_t opaque_handle) {
    int found_idx = -1;
    for (int i = 0; i < g_gpu_mapping_count; ++i) {
        if (g_gpu_mem_mappings[i].opaque_handle == opaque_handle) {
            found_idx = i;
            break;
        }
    }
    if (found_idx == -1) {
        fprintf(stderr, "PROXY ERROR: Attempted to remove non-existent mapping for opaque_handle %lu\n", opaque_handle);
        return -1; // Not found
    }
    // Shift remaining elements
    for (int i = found_idx; i < g_gpu_mapping_count - 1; ++i) {
        g_gpu_mem_mappings[i] = g_gpu_mem_mappings[i + 1];
    }
    g_gpu_mapping_count--;
    return 0;
}


// --- CUDA and IPC Initialization and Cleanup ---
int init_cuda_and_ipc(void) {
    CUresult res;
    printf("PROXY: Initializing CUDA...\n");
    res = cuInit(0);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "PROXY ERROR: cuInit failed: %d\n", res); return -1; }
    res = cuDeviceGet(&cuDevice, 0);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "PROXY ERROR: cuDeviceGet failed: %d\n", res); return -1; }
    res = cuCtxCreate(&cuContext, 0, cuDevice);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "PROXY ERROR: cuCtxCreate failed: %d\n", res); return -1; }
    printf("PROXY: CUDA initialized successfully.\n");

    printf("PROXY: Creating named pipe: %s\n", PIPE_ENCLAVE_TO_PROXY);
    if (mkfifo(PIPE_ENCLAVE_TO_PROXY, 0666) == -1 && errno != EEXIST) {
        perror("PROXY ERROR: mkfifo " PIPE_ENCLAVE_TO_PROXY); return -1;
    }
    printf("PROXY: Creating named pipe: %s\n", PIPE_PROXY_TO_ENCLAVE);
    if (mkfifo(PIPE_PROXY_TO_ENCLAVE, 0666) == -1 && errno != EEXIST) {
        perror("PROXY ERROR: mkfifo " PIPE_PROXY_TO_ENCLAVE); unlink(PIPE_ENCLAVE_TO_PROXY); return -1;
    }

    printf("PROXY: Opening pipe %s for reading...\n", PIPE_ENCLAVE_TO_PROXY);
    g_pipe_enclave_to_proxy_fd = open(PIPE_ENCLAVE_TO_PROXY, O_RDONLY);
    if (g_pipe_enclave_to_proxy_fd == -1) { perror("PROXY ERROR: open " PIPE_ENCLAVE_TO_PROXY); unlink(PIPE_ENCLAVE_TO_PROXY); unlink(PIPE_PROXY_TO_ENCLAVE); return -1; }
    printf("PROXY: Opening pipe %s for writing...\n", PIPE_PROXY_TO_ENCLAVE);
    g_pipe_proxy_to_enclave_fd = open(PIPE_PROXY_TO_ENCLAVE, O_WRONLY);
    if (g_pipe_proxy_to_enclave_fd == -1) { perror("PROXY ERROR: open " PIPE_PROXY_TO_ENCLAVE); close(g_pipe_enclave_to_proxy_fd); unlink(PIPE_ENCLAVE_TO_PROXY); unlink(PIPE_PROXY_TO_ENCLAVE); return -1; }
    printf("PROXY: Pipes opened successfully.\n");

    gpu_message_header_t init_header;
    cmd_init_req_t init_req;
    cmd_init_resp_t init_resp = {0};
    ssize_t bytes_read;

    printf("PROXY: Waiting for CMD_INIT from enclave...\n");
    bytes_read = read(g_pipe_enclave_to_proxy_fd, &init_header, sizeof(gpu_message_header_t));
    if (bytes_read != sizeof(gpu_message_header_t)) { fprintf(stderr, "PROXY ERROR: Failed to read CMD_INIT header: %s\n", strerror(errno)); return -1; }

    if (init_header.type == CMD_INIT && init_header.payload_size == sizeof(cmd_init_req_t)) {
        bytes_read = read(g_pipe_enclave_to_proxy_fd, &init_req, sizeof(cmd_init_req_t));
        if (bytes_read != sizeof(cmd_init_req_t)) { fprintf(stderr, "PROXY ERROR: Failed to read CMD_INIT payload: %s\n", strerror(errno)); return -1; }
        
        memcpy(g_proxy_session_key, init_req.session_key, sizeof(g_proxy_session_key));
        g_proxy_key_initialized = true;
        mbedtls_gcm_init(&g_proxy_aes_gcm_ctx);
        int ret = mbedtls_gcm_setkey(&g_proxy_aes_gcm_ctx, MBEDTLS_CIPHER_ID_AES, g_proxy_session_key, 256);
        if (ret != 0) {
            mbedtls_print_error("mbedtls_gcm_setkey", ret);
            // Send error response
            init_header.status = ret; // Or a generic error code
            init_resp.proxy_status_report = ret;
            send_response(g_pipe_proxy_to_enclave_fd, &init_header, &init_resp);
            return -1;
        }
        printf("PROXY: Session key received and GCM context initialized.\n");
        
        // Shared memory setup from init_req (assuming shm_path is host path, shm_size is correct)
        // For this version, g_shm_host_uri_path is predefined. In a real scenario, it might come from init_req.shm_path
        // or a config file, ensuring it matches the manifest's 'uri' for the untrusted_shm.
        g_gpu_shm_size = init_req.shm_size; // Get size from enclave's message
        printf("PROXY: Using SHM host path: %s, size: %zu\n", g_shm_host_uri_path, g_gpu_shm_size);
        int shm_fd = shm_open(g_shm_host_uri_path, O_RDWR, 0660);
        if (shm_fd == -1) { perror("PROXY ERROR: shm_open failed"); init_header.status = -errno; init_resp.proxy_status_report = -errno; send_response(g_pipe_proxy_to_enclave_fd, &init_header, &init_resp); return -1; }
        
        g_gpu_shared_mem = mmap(NULL, g_gpu_shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (g_gpu_shared_mem == MAP_FAILED) { perror("PROXY ERROR: mmap failed for SHM"); close(shm_fd); init_header.status = -errno; init_resp.proxy_status_report = -errno; send_response(g_pipe_proxy_to_enclave_fd, &init_header, &init_resp); return -1; }
        close(shm_fd);
        printf("PROXY: Shared memory mapped successfully.\n");

        g_gpu_mapping_count = 0;
        g_next_opaque_handle = 1;
        g_proxy_encryption_invocation_count = 0;

        init_header.status = 0;
        init_resp.proxy_status_report = 0;
        send_response(g_pipe_proxy_to_enclave_fd, &init_header, &init_resp);
        printf("PROXY: CMD_INIT processed, response sent.\n");
    } else {
        fprintf(stderr, "PROXY ERROR: Invalid CMD_INIT received (type: %d, payload_size: %u)\n", init_header.type, init_header.payload_size);
        return -1;
    }
    return 0;
}

void cleanup_cuda_and_ipc() {
    printf("PROXY: Cleaning up CUDA and IPC...\n");
    if (cuContext) { cuCtxDestroy(cuContext); cuContext = NULL; }
    if (g_gpu_shared_mem) { munmap(g_gpu_shared_mem, g_gpu_shm_size); g_gpu_shared_mem = NULL; }
    // Consider shm_unlink(g_shm_host_uri_path); if proxy is the one creating it.
    if (g_proxy_key_initialized) {
        mbedtls_gcm_free(&g_proxy_aes_gcm_ctx);
        memset(g_proxy_session_key, 0, sizeof(g_proxy_session_key));
        g_proxy_key_initialized = false;
    }
    if (g_pipe_enclave_to_proxy_fd != -1) { close(g_pipe_enclave_to_proxy_fd); unlink(PIPE_ENCLAVE_TO_PROXY); g_pipe_enclave_to_proxy_fd = -1; }
    if (g_pipe_proxy_to_enclave_fd != -1) { close(g_pipe_proxy_to_enclave_fd); unlink(PIPE_PROXY_TO_ENCLAVE); g_pipe_proxy_to_enclave_fd = -1; }
    printf("PROXY: Cleanup complete.\n");
}

// --- Command Handlers ---
int handle_malloc_device_cmd(const cmd_malloc_device_req_t* req, cmd_malloc_device_resp_t* resp) {
    if (!g_proxy_key_initialized) { fprintf(stderr, "PROXY ERROR: Malloc attempt before key init.\n"); return -EACCES; }
    CUdeviceptr dptr;
    CUresult res = cuMemAlloc(&dptr, req->size);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "PROXY ERROR: cuMemAlloc failed: %d\n", res); return res; }
    resp->device_ptr_opaque = add_mapping(dptr, req->size);
    if (resp->device_ptr_opaque == 0) { // Failed to add mapping
        cuMemFree(dptr); // Rollback CUDA allocation
        fprintf(stderr, "PROXY ERROR: Failed to add mapping for new allocation.\n");
        return -ENOMEM; // Or another suitable error
    }
    printf("PROXY: cuMemAlloc size %zu, opaque_handle %lu, dptr %p\n", req->size, resp->device_ptr_opaque, (void*)dptr);
    return CUDA_SUCCESS;
}

int handle_free_device_cmd(const cmd_free_device_req_t* req) {
    if (!g_proxy_key_initialized) { fprintf(stderr, "PROXY ERROR: Free attempt before key init.\n"); return -EACCES; }
    CUdeviceptr dptr = get_dptr_from_opaque(req->device_ptr_opaque);
    if (dptr == (CUdeviceptr)0) { return -EINVAL; } // Error already logged by get_dptr_from_opaque
    CUresult res = cuMemFree(dptr);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "PROXY ERROR: cuMemFree failed for dptr %p: %d\n", (void*)dptr, res); return res; }
    remove_mapping(req->device_ptr_opaque);
    printf("PROXY: cuMemFree opaque_handle %lu, dptr %p\n", req->device_ptr_opaque, (void*)dptr);
    return CUDA_SUCCESS;
}

int handle_memcpy_h2d_cmd(const cmd_memcpy_h2d_req_t* req) {
    if (!g_proxy_key_initialized) { fprintf(stderr, "PROXY ERROR: Memcpy H2D attempt before key init.\n"); return -EACCES; }
    if (!g_gpu_shared_mem || req->shm_offset + req->size > g_gpu_shm_size) { fprintf(stderr, "PROXY ERROR: Invalid SHM access for H2D.\n"); return -EINVAL; }

    CUdeviceptr dst_dptr = get_dptr_from_opaque(req->device_ptr_opaque);
    if (dst_dptr == (CUdeviceptr)0) { return -EINVAL; }

    unsigned char* encrypted_src_in_shm = (unsigned char*)g_gpu_shared_mem + req->shm_offset;
    unsigned char* temp_plaintext_buf = (unsigned char*)malloc(req->size);
    if (!temp_plaintext_buf) { perror("PROXY ERROR: malloc for H2D temp_plaintext_buf"); return -ENOMEM; }

    int ret = mbedtls_gcm_auth_decrypt(&g_proxy_aes_gcm_ctx, req->size, req->iv, sizeof(req->iv),
                                     NULL, 0, // No AAD
                                     req->tag, sizeof(req->tag),
                                     encrypted_src_in_shm, temp_plaintext_buf);
    if (ret != 0) {
        mbedtls_print_error("mbedtls_gcm_auth_decrypt (H2D)", ret);
        free(temp_plaintext_buf);
        return ret; // mbedTLS error
    }
    printf("PROXY: H2D data decrypted from SHM.\n");

    CUresult res = cuMemcpyHtoD(dst_dptr, temp_plaintext_buf, req->size);
    free(temp_plaintext_buf);
    if (res != CUDA_SUCCESS) { fprintf(stderr, "PROXY ERROR: cuMemcpyHtoD failed: %d\n", res); return res; }
    printf("PROXY: cuMemcpyHtoD successful: %zu bytes to opaque_handle %lu.\n", req->size, req->device_ptr_opaque);
    return CUDA_SUCCESS;
}

int handle_memcpy_d2h_cmd(const cmd_memcpy_d2h_req_t* req, cmd_memcpy_d2h_resp_t* resp_payload) {
    if (!g_proxy_key_initialized) { fprintf(stderr, "PROXY ERROR: Memcpy D2H attempt before key init.\n"); return -EACCES; }
    if (!g_gpu_shared_mem || req->shm_offset + req->size > g_gpu_shm_size) { fprintf(stderr, "PROXY ERROR: Invalid SHM access for D2H.\n"); return -EINVAL; }

    CUdeviceptr src_dptr = get_dptr_from_opaque(req->device_ptr_opaque);
    if (src_dptr == (CUdeviceptr)0) { return -EINVAL; }

    unsigned char* dst_encrypted_in_shm = (unsigned char*)g_gpu_shared_mem + req->shm_offset;
    unsigned char* temp_plaintext_buf = (unsigned char*)malloc(req->size);
    if (!temp_plaintext_buf) { perror("PROXY ERROR: malloc for D2H temp_plaintext_buf"); return -ENOMEM; }

    CUresult res = cuMemcpyDtoH(temp_plaintext_buf, src_dptr, req->size);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "PROXY ERROR: cuMemcpyDtoH failed: %d\n", res);
        free(temp_plaintext_buf);
        return res;
    }
    printf("PROXY: cuMemcpyDtoH successful: %zu bytes from opaque_handle %lu.\n", req->size, req->device_ptr_opaque);

    g_proxy_encryption_invocation_count++;
    proxy_construct_iv(resp_payload->iv);

    int ret = mbedtls_gcm_crypt_and_tag(&g_proxy_aes_gcm_ctx, MBEDTLS_GCM_ENCRYPT, req->size,
                                      resp_payload->iv, sizeof(resp_payload->iv),
                                      NULL, 0, // No AAD
                                      temp_plaintext_buf, dst_encrypted_in_shm,
                                      sizeof(resp_payload->tag), resp_payload->tag);
    free(temp_plaintext_buf);
    if (ret != 0) {
        mbedtls_print_error("mbedtls_gcm_crypt_and_tag (D2H)", ret);
        return ret; // mbedTLS error
    }
    resp_payload->actual_size_written_to_shm = req->size;
    printf("PROXY: D2H data encrypted to SHM.\n");
    return CUDA_SUCCESS;
}

int handle_launch_kernel_cmd(const cmd_launch_kernel_req_t* req) {
    if (!g_proxy_key_initialized) { fprintf(stderr, "PROXY ERROR: Launch kernel before key init.\n"); return -EACCES; }

    printf("PROXY: handle_launch_kernel_cmd for kernel '%s'\n", req->kernel_name);

    // Load PTX module
    // For simplicity, assume kernel_name.ptx is in current directory or a predefined path.
    char ptx_path[512];
    snprintf(ptx_path, sizeof(ptx_path), "%s.ptx", req->kernel_name); // e.g., gemm_kernel.ptx

    FILE* fp = fopen(ptx_path, "rb");
    if (!fp) {
        fprintf(stderr, "PROXY ERROR: Failed to open PTX file '%s': %s\n", ptx_path, strerror(errno));
        return -ENOENT;
    }
    fseek(fp, 0, SEEK_END);
    size_t ptx_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char* ptx_content = (char*)malloc(ptx_size + 1);
    if (!ptx_content) { fclose(fp); perror("PROXY ERROR: malloc for PTX content"); return -ENOMEM; }
    if (fread(ptx_content, 1, ptx_size, fp) != ptx_size) {
        fclose(fp); free(ptx_content); fprintf(stderr, "PROXY ERROR: Failed to read PTX file '%s'\n", ptx_path); return -EIO;
    }
    ptx_content[ptx_size] = '\0'; // Null-terminate for safety, though PTX might not need it
    fclose(fp);

    CUmodule cuModule;
    CUfunction cuFunction;
    CUresult res;

    // JIT compile options (can be empty)
    // Refer to CUDA Driver API documentation for cuModuleLoadDataEx options
    unsigned int numOptions = 0;
    CUjit_option options[1]; // Example: options[0] = CU_JIT_LOG_VERBOSE; numOptions = 1;
    void* optionValues[1];   // Example: optionValues[0] = (void*)1;

    res = cuModuleLoadDataEx(&cuModule, ptx_content, numOptions, options, optionValues);
    free(ptx_content);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "PROXY ERROR: cuModuleLoadDataEx for '%s' failed: %d\n", req->kernel_name, res);
        return res;
    }
    printf("PROXY: PTX module '%s' loaded.\n", req->kernel_name);

    res = cuModuleGetFunction(&cuFunction, cuModule, req->kernel_name);
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "PROXY ERROR: cuModuleGetFunction for '%s' failed: %d\n", req->kernel_name, res);
        cuModuleUnload(cuModule);
        return res;
    }
    printf("PROXY: Got function handle for '%s'.\n", req->kernel_name);

    // Deserialize kernel arguments
    // For GEMM: 3 CUdeviceptr (as uint64_t opaque handles), 1 int
    if (req->serialized_args_len != (3 * sizeof(uint64_t) + sizeof(int))) {
        fprintf(stderr, "PROXY ERROR: Unexpected serialized_args_len %u for GEMM kernel.\n", req->serialized_args_len);
        cuModuleUnload(cuModule);
        return -EINVAL;
    }

    uint64_t opaque_A, opaque_B, opaque_C;
    int N_val;
    unsigned char* p_args = (unsigned char*)req->serialized_args; // Cast to avoid aliasing warning
    
    memcpy(&opaque_A, p_args, sizeof(uint64_t)); p_args += sizeof(uint64_t);
    memcpy(&opaque_B, p_args, sizeof(uint64_t)); p_args += sizeof(uint64_t);
    memcpy(&opaque_C, p_args, sizeof(uint64_t)); p_args += sizeof(uint64_t);
    memcpy(&N_val, p_args, sizeof(int));

    CUdeviceptr d_A = get_dptr_from_opaque(opaque_A);
    CUdeviceptr d_B = get_dptr_from_opaque(opaque_B);
    CUdeviceptr d_C = get_dptr_from_opaque(opaque_C);

    if (!d_A || !d_B || !d_C) {
        fprintf(stderr, "PROXY ERROR: Failed to get one or more CUdeviceptr from opaque handles for kernel launch.\n");
        cuModuleUnload(cuModule);
        return -EINVAL;
    }
    
    void* kernel_params[] = { &d_A, &d_B, &d_C, &N_val };

    printf("PROXY: Launching kernel '%s' with Grid(%u,%u,%u) Block(%u,%u,%u) SharedMem(%u bytes)\n",
           req->kernel_name, req->grid_dim_x, req->grid_dim_y, req->grid_dim_z,
           req->block_dim_x, req->block_dim_y, req->block_dim_z, req->shared_mem_bytes);

    res = cuLaunchKernel(cuFunction,
                         req->grid_dim_x, req->grid_dim_y, req->grid_dim_z,
                         req->block_dim_x, req->block_dim_y, req->block_dim_z,
                         req->shared_mem_bytes,
                         NULL, // hStream
                         kernel_params,
                         NULL); // extra
    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "PROXY ERROR: cuLaunchKernel for '%s' failed: %d\n", req->kernel_name, res);
        cuModuleUnload(cuModule);
        return res;
    }
    
    // For synchronous behavior for this example, could add cuCtxSynchronize() here.
    // res = cuCtxSynchronize();
    // if (res != CUDA_SUCCESS) { /* ... handle error ... */ }


    cuModuleUnload(cuModule);
    printf("PROXY: Kernel '%s' launched and module unloaded.\n", req->kernel_name);
    return CUDA_SUCCESS;
}


// --- Main Loop ---
int main() {
    printf("Gramine CUDA Proxy starting...\n");
    if (init_cuda_and_ipc() != 0) {
        fprintf(stderr, "PROXY FATAL: CUDA or IPC initialization failed. Exiting.\n");
        cleanup_cuda_and_ipc();
        return EXIT_FAILURE;
    }
    printf("PROXY: CUDA Proxy initialized. Entering command processing loop.\n");

    gpu_message_header_t req_header;
    gpu_message_header_t resp_header;
    unsigned char payload_buffer[sizeof(cmd_launch_kernel_req_t) > sizeof(cmd_memcpy_h2d_req_t) ? sizeof(cmd_launch_kernel_req_t) : sizeof(cmd_memcpy_h2d_req_t)];


    while (1) {
        printf("PROXY: Waiting for command header...\n");
        ssize_t bytes_read = read(g_pipe_enclave_to_proxy_fd, &req_header, sizeof(gpu_message_header_t));
        if (bytes_read == 0) { printf("PROXY: Enclave pipe closed (EOF). Exiting.\n"); break; }
        if (bytes_read != sizeof(gpu_message_header_t)) {
            if (errno == EINTR) continue;
            perror("PROXY ERROR: Failed to read command header"); break;
        }

        if (req_header.payload_size > 0) {
            if (req_header.payload_size > sizeof(payload_buffer)) {
                fprintf(stderr, "PROXY ERROR: Payload size %u too large for buffer %zu.\n", req_header.payload_size, sizeof(payload_buffer));
                // TODO: Send error response, try to recover or exit. For now, exit.
                break;
            }
            bytes_read = read(g_pipe_enclave_to_proxy_fd, payload_buffer, req_header.payload_size);
            if (bytes_read != (ssize_t)req_header.payload_size) {
                if (errno == EINTR) continue; // Potentially retry if interrupted
                perror("PROXY ERROR: Failed to read command payload"); break;
            }
        }

        resp_header.type = req_header.type;
        resp_header.payload_size = 0;
        resp_header.status = -1; // Default error
        int handler_status = -1;

        switch (req_header.type) {
            case CMD_MALLOC_DEVICE: {
                cmd_malloc_device_req_t* req = (cmd_malloc_device_req_t*)payload_buffer;
                cmd_malloc_device_resp_t resp_payload = {0};
                handler_status = handle_malloc_device_cmd(req, &resp_payload);
                resp_header.status = handler_status; 
                if (handler_status == CUDA_SUCCESS) {
                    resp_header.payload_size = sizeof(cmd_malloc_device_resp_t);
                    send_response(g_pipe_proxy_to_enclave_fd, &resp_header, &resp_payload);
                } else { send_response(g_pipe_proxy_to_enclave_fd, &resp_header, NULL); }
                break;
            }
            case CMD_FREE_DEVICE: {
                cmd_free_device_req_t* req = (cmd_free_device_req_t*)payload_buffer;
                handler_status = handle_free_device_cmd(req);
                resp_header.status = handler_status; 
                send_response(g_pipe_proxy_to_enclave_fd, &resp_header, NULL);
                break;
            }
            case CMD_MEMCPY_H2D: {
                cmd_memcpy_h2d_req_t* req = (cmd_memcpy_h2d_req_t*)payload_buffer;
                handler_status = handle_memcpy_h2d_cmd(req);
                resp_header.status = handler_status; 
                send_response(g_pipe_proxy_to_enclave_fd, &resp_header, NULL);
                break;
            }
            case CMD_MEMCPY_D2H: {
                cmd_memcpy_d2h_req_t* req = (cmd_memcpy_d2h_req_t*)payload_buffer;
                cmd_memcpy_d2h_resp_t resp_payload = {0};
                handler_status = handle_memcpy_d2h_cmd(req, &resp_payload);
                resp_header.status = handler_status; 
                if (handler_status == CUDA_SUCCESS) {
                    resp_header.payload_size = sizeof(cmd_memcpy_d2h_resp_t);
                    send_response(g_pipe_proxy_to_enclave_fd, &resp_header, &resp_payload);
                } else { send_response(g_pipe_proxy_to_enclave_fd, &resp_header, NULL); }
                break;
            }
            case CMD_LAUNCH_KERNEL: {
                cmd_launch_kernel_req_t* req = (cmd_launch_kernel_req_t*)payload_buffer;
                handler_status = handle_launch_kernel_cmd(req);
                resp_header.status = handler_status; 
                send_response(g_pipe_proxy_to_enclave_fd, &resp_header, NULL);
                break;
            }
            case CMD_SHUTDOWN:
                printf("PROXY: Shutdown command received.\n");
                handler_status = 0;
                resp_header.status = handler_status; 
                send_response(g_pipe_proxy_to_enclave_fd, &resp_header, NULL);
                goto exit_loop;
            default:
                fprintf(stderr, "PROXY ERROR: Unknown command type: %d\n", req_header.type);
                handler_status = -EINVAL;
                resp_header.status = handler_status; 
                send_response(g_pipe_proxy_to_enclave_fd, &resp_header, NULL);
        }
        printf("PROXY: Command %d processed with status %d.\n", req_header.type, handler_status);
    }
exit_loop:
    cleanup_cuda_and_ipc();
    printf("Gramine CUDA Proxy exiting.\n");
    return EXIT_SUCCESS;
}
