#include "orchestrator.h"
#include "protocol.h" // Contains message type definitions
#include <iostream>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h> // For htonl/ntohl
#include <openssl/err.h>
#include <signal.h>     // For signal handling (graceful shutdown)

#include <cuda.h> 
#include <sys/mman.h>   // For mmap, munmap
#include <fcntl.h>      // For O_RDWR, shm_open (if used, though direct path mmap is simpler for this task)
#include <sys/stat.h>   // For fstat, mode constants
#include <iomanip>      // For std::hex, std::setw, std::setfill in logging

// Global flag for shutdown
std::atomic<bool> g_orchestrator_shutdown_flag(false);

void orchestrator_signal_handler(int signum) {
    std::cout << "Orchestrator: Signal " << signum << " received, initiating shutdown sequence." << std::endl;
    g_orchestrator_shutdown_flag = true;
}

// Helper to print OpenSSL errors
void print_orchestrator_ssl_errors(const std::string& prefix) {
    unsigned long err;
    while ((err = ERR_get_error()) != 0) {
        std::cerr << prefix << ": " << ERR_reason_error_string(err) << std::endl;
    }
}


// Static helper to get CUDA error strings
const char* GpuOrchestrator::get_cuda_error_string(CUresult res) {
    const char* err_str;
    if (cuGetErrorString(res, &err_str) == CUDA_SUCCESS) {
        return err_str;
    }
    return "Unknown CUDA Error";
}


GpuOrchestrator::GpuOrchestrator(int port, const char* cert_path, const char* key_path)
    : port_(port), cert_path_(cert_path), key_path_(key_path),
      server_socket_(-1), ssl_ctx_(nullptr), running_(false),
      next_gpu_mem_handle_(1), cuda_device_(0), cuda_context_(nullptr), cuda_module_(nullptr) {
    if (!init_openssl()) {
        std::cerr << "Orchestrator: OpenSSL initialization failed." << std::endl;
        // Consider throwing or setting an error state
    }
    // CUDA primary context might be initialized later per GPU, or one default here.
    // For now, init_cuda() handles basic cuInit. Context creation is per device.
}

GpuOrchestrator::~GpuOrchestrator() {
    stop_server(); // Ensure server is stopped
    
    // Clean up CUDA resources
    // This should ideally iterate through all created contexts and free resources.
    // For simplicity, cleaning up based on single device/context members.
    {
        std::lock_guard<std::mutex> lock_ctx(cuda_context_mutex_); // Protect context ops
        if (cuda_context_) {
            std::cout << "Orchestrator: Destroying CUDA context for device " << cuda_device_ << std::endl;
            // Free all allocated memory associated with this context before destroying it
            {
                std::lock_guard<std::mutex> lock_mem(gpu_mem_mutex_);
                for (auto const& [handle, dptr] : allocated_gpu_mem_) {
                    std::cout << "Orchestrator: Auto-freeing GPU memory handle " << handle << std::endl;
                    cuMemFree_v2(dptr);
                }
                allocated_gpu_mem_.clear();
            }
            if (cuda_module_) {
                 std::cout << "Orchestrator: Unloading CUDA module." << std::endl;
                 cuModuleUnload(cuda_module_);
                 cuda_module_ = nullptr;
                 loaded_kernels_.clear();
            }
            cuCtxDestroy_v2(cuda_context_);
            cuda_context_ = nullptr;
        }
    }


    if (ssl_ctx_) {
        SSL_CTX_free(ssl_ctx_);
        ssl_ctx_ = nullptr;
    }
    cleanup_openssl();
    std::cout << "Orchestrator: GPU Orchestrator shut down." << std::endl;
}

bool GpuOrchestrator::init_cuda() {
    CUresult cu_res = cuInit(0);
    if (cu_res != CUDA_SUCCESS) {
        std::cerr << "Orchestrator: cuInit(0) failed: " << get_cuda_error_string(cu_res) << std::endl;
        return false;
    }
    std::cout << "Orchestrator: CUDA initialized successfully (cuInit)." << std::endl;

    // Get device handle (assuming device 0 for this simple orchestrator)
    // In a multi-GPU setup, this might be done per requested gpu_id.
    cu_res = cuDeviceGet(&cuda_device_, 0); 
    if (cu_res != CUDA_SUCCESS) {
        std::cerr << "Orchestrator: cuDeviceGet for device 0 failed: " << get_cuda_error_string(cu_res) << std::endl;
        return false;
    }
    char device_name[256];
    cuDeviceGetName(device_name, sizeof(device_name), cuda_device_);
    std::cout << "Orchestrator: Defaulting to CUDA Device 0: " << device_name << std::endl;
    return true;
}

bool GpuOrchestrator::init_openssl() {
    SSL_library_init();
    OpenSSL_add_all_algorithms();
    SSL_load_error_strings();
    ERR_load_BIO_strings();
    ERR_load_crypto_strings();
    return true;
}

void GpuOrchestrator::cleanup_openssl() {
    ERR_free_strings();
    EVP_cleanup();
}

SSL_CTX* GpuOrchestrator::create_ssl_context() {
    const SSL_METHOD* method;
    SSL_CTX* ctx;

    method = TLS_server_method();
    ctx = SSL_CTX_new(method);
    if (!ctx) {
        std::cerr << "Orchestrator: Error creating SSL context." << std::endl;
        print_orchestrator_ssl_errors("SSL_CTX_new");
        return nullptr;
    }
    // Set min TLS version
    if (SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION) == 0) {
        std::cerr << "Orchestrator: Error setting min TLS version." << std::endl;
        print_orchestrator_ssl_errors("SSL_CTX_set_min_proto_version");
        SSL_CTX_free(ctx);
        return nullptr;
    }
    return ctx;
}

void GpuOrchestrator::configure_ssl_context(SSL_CTX* ctx) {
    // Instructions for generating orchestrator.crt and orchestrator.key:
    // openssl req -x509 -newkey rsa:4096 -keyout orchestrator.key -out orchestrator.crt -days 365 -nodes -subj "/CN=gpu-orchestrator"
    if (SSL_CTX_use_certificate_file(ctx, cert_path_, SSL_FILETYPE_PEM) <= 0) {
        std::cerr << "Orchestrator: Error loading certificate file: " << cert_path_ << std::endl;
        print_orchestrator_ssl_errors("SSL_CTX_use_certificate_file");
        return; // Or throw
    }
    if (SSL_CTX_use_PrivateKey_file(ctx, key_path_, SSL_FILETYPE_PEM) <= 0) {
        std::cerr << "Orchestrator: Error loading private key file: " << key_path_ << std::endl;
        print_orchestrator_ssl_errors("SSL_CTX_use_PrivateKey_file");
        return; // Or throw
    }
    if (!SSL_CTX_check_private_key(ctx)) {
        std::cerr << "Orchestrator: Private key does not match the public certificate." << std::endl;
        print_orchestrator_ssl_errors("SSL_CTX_check_private_key");
        return; // Or throw
    }
    std::cout << "Orchestrator: Certificate and private key loaded successfully." << std::endl;
}

void GpuOrchestrator::start_server() {
    if (running_) {
        std::cout << "Orchestrator: Server is already running." << std::endl;
        return;
    }

    ssl_ctx_ = create_ssl_context();
    if (!ssl_ctx_) return;
    configure_ssl_context(ssl_ctx_);
    // A more robust check would be needed here if configure_ssl_context could fail and not exit/throw
    // For example, check if cert/key are actually loaded in ctx.

    server_socket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket_ == -1) {
        std::cerr << "Orchestrator: Error creating socket: " << strerror(errno) << std::endl;
        return;
    }

    // Allow address reuse
    int opt = 1;
    if (setsockopt(server_socket_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        std::cerr << "Orchestrator: setsockopt(SO_REUSEADDR) failed: " << strerror(errno) << std::endl;
        close(server_socket_);
        server_socket_ = -1;
        return;
    }


    sockaddr_in server_address{};
    server_address.sin_family = AF_INET;
    server_address.sin_addr.s_addr = INADDR_ANY;
    server_address.sin_port = htons(port_);

    if (bind(server_socket_, (struct sockaddr*)&server_address, sizeof(server_address)) < 0) {
        std::cerr << "Orchestrator: Error binding socket: " << strerror(errno) << std::endl;
        close(server_socket_);
        server_socket_ = -1;
        return;
    }

    if (listen(server_socket_, 10) < 0) { // Backlog of 10
        std::cerr << "Orchestrator: Error listening on socket: " << strerror(errno) << std::endl;
        close(server_socket_);
        server_socket_ = -1;
        return;
    }

    running_ = true;
    accept_thread_ = std::thread(&GpuOrchestrator::accept_connections, this);
    std::cout << "Orchestrator: Server started on port " << port_ << ", waiting for enclave connections..." << std::endl;
}

void GpuOrchestrator::stop_server() {
    if (!running_) return;

    running_ = false;
    g_orchestrator_shutdown_flag = true; // Signal global shutdown

    if (server_socket_ != -1) {
        // Shutdown the listening socket to unblock accept()
        shutdown(server_socket_, SHUT_RDWR); // Or SHUT_RD
        // close(server_socket_); // Close it after thread joins
    }

    if (accept_thread_.joinable()) {
        accept_thread_.join();
    }
    
    if (server_socket_ != -1) {
        close(server_socket_);
        server_socket_ = -1;
    }

    std::cout << "Orchestrator: Server stopped." << std::endl;
    // Note: Active client handler threads are not explicitly managed/joined here for simplicity.
    // In a production server, you'd need a mechanism to signal them to stop and join them.
}

void GpuOrchestrator::accept_connections() {
    while (running_ && !g_orchestrator_shutdown_flag) {
        sockaddr_in client_address{};
        socklen_t client_len = sizeof(client_address);
        int client_socket = accept(server_socket_, (struct sockaddr*)&client_address, &client_len);

        if (g_orchestrator_shutdown_flag || !running_) {
            if (client_socket >= 0) close(client_socket);
            break;
        }

        if (client_socket < 0) {
            if (errno == EINTR && g_orchestrator_shutdown_flag) {
                 std::cout << "Orchestrator: Accept interrupted by shutdown signal." << std::endl;
            } else if (running_) { // Don't print error if accept failed due to stop_server() closing the socket
                 std::cerr << "Orchestrator: Error accepting client connection: " << strerror(errno) << std::endl;
            }
            continue;
        }

        std::cout << "Orchestrator: Enclave connected (raw socket: " << client_socket << "). Performing TLS handshake..." << std::endl;

        SSL* ssl = SSL_new(ssl_ctx_);
        if (!ssl) {
            std::cerr << "Orchestrator: Error creating SSL structure for new connection." << std::endl;
            print_orchestrator_ssl_errors("SSL_new");
            close(client_socket);
            continue;
        }
        SSL_set_fd(ssl, client_socket);

        if (SSL_accept(ssl) <= 0) {
            std::cerr << "Orchestrator: Error performing SSL handshake with enclave. Socket: " << client_socket << std::endl;
            print_orchestrator_ssl_errors("SSL_accept");
            SSL_free(ssl);
            close(client_socket);
            continue;
        }
        std::cout << "Orchestrator: TLS handshake successful with enclave (socket: " << client_socket << ")" << std::endl;

        // Create a new thread to handle this enclave connection
        std::thread handler_thread(&GpuOrchestrator::handle_enclave_connection, this, client_socket, ssl);
        handler_thread.detach(); // Detach as we're not managing them explicitly for now
    }
    std::cout << "Orchestrator: Accept connections loop finished." << std::endl;
}


bool GpuOrchestrator::receive_enclave_quote(SSL* ssl, std::vector<uint8_t>& quote_vec) {
    uint32_t quote_size_net;
    int bytes_read = SSL_read(ssl, &quote_size_net, sizeof(quote_size_net));
    if (bytes_read <= 0) {
        std::cerr << "Orchestrator: Error receiving enclave quote size." << std::endl;
        print_orchestrator_ssl_errors("SSL_read enclave quote size");
        return false;
    }
    if (bytes_read != sizeof(quote_size_net)) {
         std::cerr << "Orchestrator: Received incomplete enclave quote size." << std::endl;
         return false;
    }

    uint32_t quote_size = ntohl(quote_size_net);
    if (quote_size == 0 || quote_size > 16384) { // Sanity check (e.g., max 16KB)
        std::cerr << "Orchestrator: Invalid enclave quote size received: " << quote_size << std::endl;
        return false;
    }

    quote_vec.resize(quote_size);
    bytes_read = 0;
    int total_received = 0;
    while(total_received < (int)quote_size) {
        bytes_read = SSL_read(ssl, quote_vec.data() + total_received, quote_size - total_received);
        if (bytes_read <= 0) {
            std::cerr << "Orchestrator: Error receiving enclave quote data. Received " << total_received << " of " << quote_size << std::endl;
            print_orchestrator_ssl_errors("SSL_read enclave quote data");
            return false;
        }
        total_received += bytes_read;
    }
    
    if (total_received != (int)quote_size) {
        std::cerr << "Orchestrator: Enclave quote data reception incomplete. Expected " << quote_size << " got " << total_received << std::endl;
        return false;
    }
    std::cout << "Orchestrator: Enclave quote received. Size: " << quote_size << " bytes." << std::endl;
    return true;
}

bool GpuOrchestrator::verify_enclave_quote(const uint8_t* quote_data, uint32_t quote_size) {
    std::cout << "Orchestrator: Verifying enclave quote (Placeholder)... Size: " << quote_size << " bytes." << std::endl;
    // TODO: Implement actual enclave quote verification using libsgx_dcap_quoteverify or similar.
    // This involves:
    // 1. Calling sgx_qv_get_quote_supplemental_data_size() and sgx_qv_verify_quote().
    // 2. Checking the verification result and potentially collateral expiration status.
    // 3. Extracting and validating the report_data from the quote (e.g., ensuring it contains a nonce
    //    or hash of the orchestrator's TLS certificate to prevent replay attacks).
    std::cout << "Orchestrator: Enclave quote verification placeholder: Succeeded." << std::endl;
    return true; // For now, always return true.
}

bool GpuOrchestrator::orchestrator_read_message(SSL* ssl, uint8_t& msg_type, std::vector<uint8_t>& payload) {
    // 1. Read message type (1 byte)
    int bytes_read = SSL_read(ssl, &msg_type, 1);
    if (bytes_read <= 0) {
        if (bytes_read == 0) std::cout << "Orchestrator: Enclave disconnected while reading message type." << std::endl;
        else {
            std::cerr << "Orchestrator: Error reading message type from enclave." << std::endl;
            print_orchestrator_ssl_errors("SSL_read msg_type");
        }
        return false;
    }

    // 2. Read payload length (4 bytes, network byte order)
    uint32_t payload_len_net;
    bytes_read = SSL_read(ssl, &payload_len_net, sizeof(payload_len_net));
    if (bytes_read <= 0) {
        if (bytes_read == 0) std::cout << "Orchestrator: Enclave disconnected while reading payload length." << std::endl;
        else {
            std::cerr << "Orchestrator: Error reading payload length from enclave." << std::endl;
            print_orchestrator_ssl_errors("SSL_read payload_len");
        }
        return false;
    }
     if (bytes_read != sizeof(payload_len_net)) {
         std::cerr << "Orchestrator: Received incomplete payload length from enclave." << std::endl;
         return false;
    }
    uint32_t payload_len = ntohl(payload_len_net);

    const uint32_t MAX_PAYLOAD_SIZE = 1024 * 1024 * 10; // Example: 10MB limit for GPU data
    if (payload_len > MAX_PAYLOAD_SIZE) {
        std::cerr << "Orchestrator: Payload length " << payload_len << " exceeds max " << MAX_PAYLOAD_SIZE << "." << std::endl;
        return false; 
    }
    
    if (payload_len > 0) {
        payload.resize(payload_len);
        bytes_read = 0;
        uint32_t total_payload_read = 0;
        while (total_payload_read < payload_len) {
            bytes_read = SSL_read(ssl, payload.data() + total_payload_read, payload_len - total_payload_read);
            if (bytes_read <= 0) {
                 if (bytes_read == 0) std::cout << "Orchestrator: Enclave disconnected while reading payload." << std::endl;
                 else {
                    std::cerr << "Orchestrator: Error reading payload from enclave." << std::endl;
                    print_orchestrator_ssl_errors("SSL_read payload data");
                 }
                return false;
            }
            total_payload_read += bytes_read;
        }
    } else {
        payload.clear();
    }
    return true;
}

bool GpuOrchestrator::orchestrator_send_response(SSL* ssl, uint8_t msg_type, const std::vector<uint8_t>& payload) {
    // 1. Send message type (1 byte)
    int bytes_sent = SSL_write(ssl, &msg_type, 1);
    if (bytes_sent <= 0) {
        std::cerr << "Orchestrator: Error sending response type to enclave." << std::endl;
        print_orchestrator_ssl_errors("SSL_write msg_type");
        return false;
    }

    // 2. Send payload length (4 bytes, network byte order)
    uint32_t payload_len_net = htonl(static_cast<uint32_t>(payload.size()));
    bytes_sent = SSL_write(ssl, &payload_len_net, sizeof(payload_len_net));
    if (bytes_sent <= 0) {
        std::cerr << "Orchestrator: Error sending payload length to enclave." << std::endl;
        print_orchestrator_ssl_errors("SSL_write payload_len");
        return false;
    }
     if (bytes_sent != sizeof(payload_len_net)) {
         std::cerr << "Orchestrator: Sent incomplete payload length to enclave." << std::endl;
         return false;
    }

    // 3. Send payload data (if any)
    if (!payload.empty()) {
        bytes_sent = SSL_write(ssl, payload.data(), payload.size());
        if (bytes_sent <= 0 || (size_t)bytes_sent != payload.size()) {
            std::cerr << "Orchestrator: Error sending payload data to enclave." << std::endl;
            print_orchestrator_ssl_errors("SSL_write payload data");
            return false;
        }
    }
    return true;
}


void GpuOrchestrator::handle_enclave_connection(int client_socket, SSL* ssl) {
    std::cout << "Orchestrator: Handling new enclave connection. Socket: " << client_socket << std::endl;

    // 1. Receive and verify enclave's SGX quote
    std::vector<uint8_t> enclave_quote_vec;
    if (!receive_enclave_quote(ssl, enclave_quote_vec)) {
        std::cerr << "Orchestrator: Failed to receive enclave quote. Closing connection. Socket: " << client_socket << std::endl;
        goto cleanup_connection;
    }

    if (!verify_enclave_quote(enclave_quote_vec.data(), enclave_quote_vec.size())) {
        std::cerr << "Orchestrator: Enclave quote verification failed. Closing connection. Socket: " << client_socket << std::endl;
        // Optionally send a NACK or specific error message before closing
        orchestrator_send_response(ssl, GPU_ORCH_RESP_NACK, {'Q', 'V', 'F'}); // QV_Failed
        goto cleanup_connection;
    }
    std::cout << "Orchestrator: Enclave attestation successful. Socket: " << client_socket << std::endl;
    // Optionally send an attestation success ACK to the enclave if protocol requires it
    // For now, proceed directly to message loop.

    // 2. IPC Message Handling Loop
    // Ensure CUDA context is pushed for this thread before entering the loop
    // This depends on how contexts are managed (per-thread, per-device global, etc.)
    // For a simple single-device, single-context model used here, we ensure it's set after INIT_DEVICE.

    while (running_ && !g_orchestrator_shutdown_flag) {
        uint8_t msg_type;
        std::vector<uint8_t> payload;

        if (!orchestrator_read_message(ssl, msg_type, payload)) {
            std::cout << "Orchestrator: Enclave disconnected or read_message failed. Socket: " << client_socket << std::endl;
            break; 
        }

        std::cout << "Orchestrator: Received message from enclave. Type: 0x" << std::hex << (int)msg_type 
                  << std::dec << ", Payload size: " << payload.size() << ". Socket: " << client_socket << std::endl;
        
        CUresult cu_res; // Declare here for use in switch cases

        switch (msg_type) {
            case GPU_ORCH_CMD_INIT_DEVICE: {
                if (payload.size() < sizeof(uint32_t)) {
                    orchestrator_send_response(ssl, GPU_ORCH_RESP_NACK, std::vector<uint8_t>(std::string("BAD_PAYLOAD_INIT_DEV").begin(), std::string("BAD_PAYLOAD_INIT_DEV").end()));
                    continue;
                }
                uint32_t gpu_id;
                memcpy(&gpu_id, payload.data(), sizeof(uint32_t));
                
                if (gpu_id != 0) { 
                    std::cerr << "Orchestrator: INIT_DEVICE requested for non-zero GPU ID " << gpu_id << ", only device 0 is supported by this orchestrator." << std::endl;
                    orchestrator_send_response(ssl, GPU_ORCH_RESP_NACK, std::vector<uint8_t>(std::string("UNSUPPORTED_GPU_ID").begin(), std::string("UNSUPPORTED_GPU_ID").end()));
                    continue;
                }
                std::cout << "Orchestrator: Processing INIT_DEVICE for GPU ID " << gpu_id << std::endl;
                
                std::string nack_reason;
                bool init_success = false;
                { // Scoping for cuda_context_mutex_
                    std::lock_guard<std::mutex> lock(cuda_context_mutex_);
                    if (!cuda_context_) { 
                        cu_res = cuDeviceGet(&cuda_device_, gpu_id); // Re-get based on requested ID, though here it's always 0
                        if (cu_res != CUDA_SUCCESS) {
                            nack_reason = std::string("DEV_GET_FAIL:") + get_cuda_error_string(cu_res);
                        } else {
                            cu_res = cuCtxCreate(&cuda_context_, 0, cuda_device_);
                            if (cu_res != CUDA_SUCCESS) {
                                nack_reason = std::string("CTX_CREATE_FAIL:") + get_cuda_error_string(cu_res);
                            } else {
                                std::cout << "Orchestrator: CUDA Context created for device " << cuda_device_ << std::endl;
                                // Load predefined module (e.g., vector_add.ptx)
                                const char* ptx_path = "vector_add.ptx"; 
                                cu_res = cuModuleLoad(&cuda_module_, ptx_path);
                                if (cu_res != CUDA_SUCCESS) {
                                    std::cerr << "Orchestrator: cuModuleLoad of " << ptx_path << " failed: " << get_cuda_error_string(cu_res) << std::endl;
                                    cuda_module_ = nullptr; 
                                    // Not critical for init, enclave can request kernel load later or this can be optional
                                } else {
                                    std::cout << "Orchestrator: CUDA module '" << ptx_path << "' loaded." << std::endl;
                                    CUfunction kernel_func;
                                    const char* kernel_name = "vector_add_kernel";
                                    cu_res = cuModuleGetFunction(&kernel_func, cuda_module_, kernel_name);
                                    if (cu_res == CUDA_SUCCESS) {
                                        std::lock_guard<std::mutex> kernel_lock(loaded_kernels_mutex_);
                                        loaded_kernels_[kernel_name] = kernel_func;
                                        std::cout << "Orchestrator: Kernel '" << kernel_name << "' loaded from module." << std::endl;
                                    } else {
                                         std::cerr << "Orchestrator: cuModuleGetFunction for '" << kernel_name << "' failed: " << get_cuda_error_string(cu_res) << std::endl;
                                    }
                                }
                            }
                        }
                    } // else context already exists

                    if (cuda_context_) { // If context exists or was just created successfully
                        cu_res = cuCtxSetCurrent(cuda_context_); // Set for this handler thread
                        if (cu_res != CUDA_SUCCESS) {
                             nack_reason = std::string("CTX_SET_FAIL:") + get_cuda_error_string(cu_res);
                        } else {
                            init_success = true;
                        }
                    }
                } // Mutex released
                
                if(init_success) {
                    orchestrator_send_response(ssl, GPU_ORCH_RESP_ACK);
                } else {
                    std::cerr << "Orchestrator: INIT_DEVICE failed. Reason: " << (nack_reason.empty() ? "Unknown" : nack_reason) << std::endl;
                    orchestrator_send_response(ssl, GPU_ORCH_RESP_NACK, std::vector<uint8_t>(nack_reason.begin(), nack_reason.end()));
                }
                break;
            }
            // Other cases will be added in subsequent diffs
            default:
                std::cerr << "Orchestrator: Unknown message type from enclave: 0x" << std::hex << (int)msg_type << std::dec << std::endl;
                orchestrator_send_response(ssl, GPU_ORCH_RESP_NACK, std::vector<uint8_t>(std::string("UNKNOWN_MSG_TYPE").begin(), std::string("UNKNOWN_MSG_TYPE").end()));
                break;
        }
    }

cleanup_connection:
    if (ssl) {
        SSL_shutdown(ssl); // Attempt graceful SSL shutdown
        SSL_free(ssl);
    }
    if (client_socket != -1) {
        close(client_socket);
    }
    std::cout << "Orchestrator: Enclave connection closed and resources freed. Socket: " << client_socket << std::endl;
}
