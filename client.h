#ifndef CLIENT_H
#define CLIENT_H

#include <string>
#include <vector>
#include <openssl/ssl.h>

#include <string>
#include <vector>
#include <openssl/ssl.h>
#include <openssl/err.h> // For error handling if used directly
#include <openssl/sha.h> // For hashing, if used for report data

// Assuming protocol.h is available for message types, not directly used in attestation flow but for app layer
// #include "protocol.h" 

// Basic SGX types (can be replaced with actual sgx_urts.h or sgx_report.h if available/needed by toolchain)
#ifndef SGX_REPORT_DATA_SIZE
#define SGX_REPORT_DATA_SIZE 64
#endif
typedef struct _sgx_report_data_t {
    unsigned char d[SGX_REPORT_DATA_SIZE];
} sgx_report_data_t;


class Client {
public:
    Client(const std::string& server_ip, int server_port, const std::string& client_id);
    ~Client();

    bool connect_and_attest(); // Combines connection, TLS, and attestation
    void run_application_protocol(); 
    void disconnect();

private:
    // TLS specific methods
    bool init_openssl();
    void cleanup_openssl();
    SSL_CTX* create_ssl_context();
    bool perform_tls_handshake();

    // SGX Attestation specific methods
    bool perform_sgx_attestation_flow(); // Renamed from perform_sgx_attestation for clarity
    bool generate_client_quote(std::vector<uint8_t>& client_quote_vec, const sgx_report_data_t& report_data);
    bool send_client_attestation_data(SSL* ssl, const std::vector<uint8_t>& client_quote);
    bool receive_server_attestation_data(SSL* ssl, std::vector<uint8_t>& server_quote_vec); // Renamed parameter for consistency
    bool verify_server_quote(const uint8_t* quote_data, uint32_t quote_size); // Placeholder

    // Helper for SGX attestation file I/O (adapted from server if client is in Gramine)
    // These are illustrative and their implementation details depend on whether the client runs in Gramine.
    bool client_write_to_attestation_file(const char* filepath, const void* data, size_t count);
    std::vector<uint8_t> client_read_from_attestation_file(const char* filepath, size_t read_len = 0);
    // Helper to prepare report data, e.g. with a nonce.
    // get_tls_certificate_hash is less relevant if client uses nonce for its quote's report data.
    void prepare_client_report_data(sgx_report_data_t& report_data);

    // Client-side message helpers
    bool client_send_message(SSL* ssl, uint8_t msg_type, const std::vector<uint8_t>& payload = {});
    bool client_receive_message(SSL* ssl, uint8_t& msg_type, std::vector<uint8_t>& payload);

    // Application protocol specific methods
    bool register_self(); // Renamed, client_id_ is member
    bool send_data_to_recipient(const std::string& recipient_id, const std::vector<uint8_t>& data);
    bool poll_for_data();
    // handle_server_response is implicitly part of the above methods now

    // Network/SSL state
    std::string server_ip_;
    int server_port_;
    int client_socket_;
    SSL_CTX* ssl_ctx_;
    SSL* ssl_;
    std::string client_id_; // ID for this client instance
};

#endif // CLIENT_H
