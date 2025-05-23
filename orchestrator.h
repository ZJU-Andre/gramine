#ifndef ORCHESTRATOR_H
#define ORCHESTRATOR_H

#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <openssl/ssl.h>

// Forward declaration for CUDA, actual include in .cpp
// typedef int CUresult; // Example, actual types might be more complex

class GpuOrchestrator {
public:
    GpuOrchestrator(int port, const char* cert_path, const char* key_path);
    ~GpuOrchestrator();

    bool init_cuda(); // Basic CUDA initialization
    void start_server();
    void stop_server();

private:
    // Network and TLS
    bool init_openssl();
    void cleanup_openssl();
    SSL_CTX* create_ssl_context();
    void configure_ssl_context(SSL_CTX* ctx);
    void accept_connections();
    void handle_enclave_connection(int client_socket, SSL* ssl); // Renamed from handle_client for clarity

    // Attestation (Placeholder)
    bool receive_enclave_quote(SSL* ssl, std::vector<uint8_t>& quote_vec);
    bool verify_enclave_quote(const uint8_t* quote_data, uint32_t quote_size); // Placeholder

    // IPC Message Handling
    bool orchestrator_read_message(SSL* ssl, uint8_t& msg_type, std::vector<uint8_t>& payload);
    bool orchestrator_send_response(SSL* ssl, uint8_t msg_type, const std::vector<uint8_t>& payload = {});

    // Member variables
    int port_;
    const char* cert_path_;
    const char* key_path_;
    int server_socket_;
    SSL_CTX* ssl_ctx_;
    std::atomic<bool> running_;
    std::thread accept_thread_;
    // Potentially a list of client handler threads if we need to manage/join them later
};

#endif // ORCHESTRATOR_H
