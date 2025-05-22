#ifndef SERVER_H
#define SERVER_H

#include <atomic>
#include <thread>
#include <vector>
#include <string>
#include <queue>
#include <map>
#include <mutex>
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/sha.h>

#include "protocol.h" // For message type constants

// SGX Remote Attestation (DCAP) related:
// Ensure the Gramine manifest will include:
// sgx.remote_attestation = "dcap"
// sgx.allowed_files = [
//  { uri = "file:/dev/attestation/user_report_data", sgx_trusted = true },
//  { uri = "file:/dev/attestation/quote", sgx_trusted = true },
//  // The following might be needed if implementing client quote verification fully,
//  // or if the server needs to get target_info for other purposes.
//  // { uri = "file:/dev/attestation/target_info", sgx_trusted = true },
//  // { uri = "file:/dev/attestation/report", sgx_trusted = true }
// ]
// It's often simpler to use:
// sgx.trusted_files = [ "file:/dev/attestation/" ]
// Check Gramine documentation for the most up-to-date manifest requirements for DCAP.

// Basic SGX types (can be replaced with actual sgx_report.h if available/needed)
// These are typically 64 bytes.
#ifndef SGX_REPORT_DATA_SIZE
#define SGX_REPORT_DATA_SIZE 64
#endif
typedef struct _sgx_report_data_t {
    unsigned char d[SGX_REPORT_DATA_SIZE];
} sgx_report_data_t;


class Server {
public:
    Server(int port, const char* cert_path, const char* key_path);
    ~Server();

    void start();
    void stop();
    bool is_running() const { return running_; }

private:
    void init_openssl();
    void cleanup_openssl();
    SSL_CTX* create_ssl_context();
    void configure_ssl_context(SSL_CTX* ctx);

    void accept_connections();
    void handle_client_tls(SSL* ssl);

private: 
    // Attestation helper methods
    bool get_tls_certificate_hash(SSL* ssl, sgx_report_data_t& report_data);
    bool write_to_attestation_file(const char* filepath, const void* data, size_t count);
    std::vector<uint8_t> read_from_attestation_file(const char* filepath, size_t read_len = 0); // read_len=0 means read till EOF

    // Client-side attestation handling
    bool receive_client_attestation_data(SSL* ssl, std::vector<uint8_t>& client_quote);
    bool verify_client_quote(const uint8_t* quote_data, uint32_t quote_size);

    // Server-side attestation generation and sending
    bool generate_server_quote(SSL* ssl_for_cert_hash, std::vector<uint8_t>& server_quote_vec);
    bool send_server_attestation_data(SSL* ssl, const std::vector<uint8_t>& server_quote);

    // Application-level messaging helpers
    bool read_message(SSL* ssl, uint8_t& msg_type, std::vector<uint8_t>& payload);
    bool send_response(SSL* ssl, uint8_t msg_type, const std::vector<uint8_t>& payload = {});
    bool send_simple_response(SSL* ssl, uint8_t msg_type); // Wrapper for send_response with empty payload


    int port_;
    const char* cert_path_;
    const char* key_path_;
    int server_socket_;
    std::atomic<bool> running_;
    std::thread accept_thread_;
    SSL_CTX *ssl_ctx_;

    // Data structures for application-level messaging
    std::map<std::string, std::queue<std::vector<uint8_t>>> client_message_queues_;
    std::mutex client_message_queues_mutex_;

    std::map<SSL*, std::string> active_client_ids_; // Maps current SSL session to registered client_id
    std::mutex active_client_ids_mutex_;
};

#endif // SERVER_H
