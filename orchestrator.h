#ifndef ORCHESTRATOR_H
#define ORCHESTRATOR_H

#include <string>
#include <vector>
#include <thread>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <map>    // For std::map
#include <mutex>  // For std::mutex
#include <openssl/ssl.h>
#include <cuda.h> // Include CUDA driver API header

// Forward declaration for CUDA, actual include in .cpp
// typedef int CUresult; // No longer needed as cuda.h is included

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

    // GPU Resource Management
    std::map<uint64_t, CUdeviceptr> allocated_gpu_mem_;
    std::mutex gpu_mem_mutex_;
    uint64_t next_gpu_mem_handle_ = 1;

    // Kernel and Context Management
    // Assuming single GPU for now, these could be maps from gpu_id if multi-GPU
    CUdevice    cuda_device_ = 0; // Default to device 0
    CUcontext   cuda_context_ = nullptr; 
    std::mutex  cuda_context_mutex_; // To protect context creation/setting for operations
    
    // For simplicity, pre-load one module and kernel, or load on first init_device.
    // A more robust solution would map gpu_id to modules/kernels.
    CUmodule    cuda_module_ = nullptr; 
    std::map<std::string, CUfunction> loaded_kernels_; // Maps kernel name (string) to CUfunction
    std::mutex  loaded_kernels_mutex_;

    // Helper to get CUDA error string
    static const char* get_cuda_error_string(CUresult res);
};

#endif // ORCHESTRATOR_H
