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

// For CUDA initialization
#include <cuda.h> // This needs the CUDA toolkit to be installed

// Global flag for shutdown
std::atomic<bool> g_orchestrator_shutdown_flag(false);

void orchestrator_signal_handler(int signum) {
    std::cout << "Orchestrator: Signal " << signum << " received, shutting down." << std::endl;
    g_orchestrator_shutdown_flag = true;
}

// Helper to print OpenSSL errors
void print_orchestrator_ssl_errors(const std::string& prefix) {
    unsigned long err;
    while ((err = ERR_get_error()) != 0) {
        std::cerr << prefix << ": " << ERR_reason_error_string(err) << std::endl;
    }
}

GpuOrchestrator::GpuOrchestrator(int port, const char* cert_path, const char* key_path)
    : port_(port), cert_path_(cert_path), key_path_(key_path),
      server_socket_(-1), ssl_ctx_(nullptr), running_(false) {
    if (!init_openssl()) {
        std::cerr << "Orchestrator: OpenSSL initialization failed." << std::endl;
        // Consider throwing or setting an error state
    }
}

GpuOrchestrator::~GpuOrchestrator() {
    stop_server(); // Ensure server is stopped
    if (ssl_ctx_) {
        SSL_CTX_free(ssl_ctx_);
        ssl_ctx_ = nullptr;
    }
    cleanup_openssl();
}

bool GpuOrchestrator::init_cuda() {
    CUresult cu_res = cuInit(0);
    if (cu_res != CUDA_SUCCESS) {
        const char* err_str;
        cuGetErrorString(cu_res, &err_str);
        std::cerr << "Orchestrator: CUDA initialization failed. cuInit(0) returned " << err_str << " (" << cu_res << ")" << std::endl;
        return false;
    }
    std::cout << "Orchestrator: CUDA initialized successfully." << std::endl;
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
    while (running_ && !g_orchestrator_shutdown_flag) {
        uint8_t msg_type;
        std::vector<uint8_t> payload;

        if (!orchestrator_read_message(ssl, msg_type, payload)) {
            std::cout << "Orchestrator: Enclave disconnected or read_message failed. Socket: " << client_socket << std::endl;
            break; 
        }

        std::cout << "Orchestrator: Received message from enclave. Type: 0x" << std::hex << (int)msg_type 
                  << std::dec << ", Payload size: " << payload.size() << ". Socket: " << client_socket << std::endl;

        // TODO: Implement dispatch logic based on msg_type for GPU operations.
        switch (msg_type) {
            case GPU_ORCH_MSG_REQUEST_COMPUTE:
                std::cout << "Orchestrator: Received GPU_ORCH_MSG_REQUEST_COMPUTE." << std::endl;
                // Placeholder: process compute request
                orchestrator_send_response(ssl, GPU_ORCH_RESP_ACK, {'C', 'M', 'P', '_', 'A', 'C', 'K'});
                break;
            case GPU_ORCH_MSG_DATA_TRANSFER_TO_GPU:
                 std::cout << "Orchestrator: Received GPU_ORCH_MSG_DATA_TRANSFER_TO_GPU. Data size: " << payload.size() << std::endl;
                // Placeholder: handle data transfer to GPU
                orchestrator_send_response(ssl, GPU_ORCH_RESP_ACK, {'D', 'T', 'G', '_', 'A', 'C', 'K'});
                break;
            case GPU_ORCH_MSG_DATA_TRANSFER_FROM_GPU:
                 std::cout << "Orchestrator: Received GPU_ORCH_MSG_DATA_TRANSFER_FROM_GPU (request for data)." << std::endl;
                // Placeholder: handle data transfer from GPU request
                // This would involve getting data and then sending GPU_ORCH_RESP_DATA_READY
                orchestrator_send_response(ssl, GPU_ORCH_RESP_DATA_READY, {'S', 'A', 'M', 'P', 'L', 'E'}); // Send some sample data
                break;
            default:
                std::cerr << "Orchestrator: Unknown message type from enclave: 0x" << std::hex << (int)msg_type << std::dec << std::endl;
                orchestrator_send_response(ssl, GPU_ORCH_RESP_NACK, {'U', 'N', 'K'}); // Unknown type
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
