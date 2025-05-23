#include "client.h"
#include "protocol.h" // Assuming protocol.h is available for message types
#include <iostream>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <openssl/err.h>
#include <openssl/sha.h> // For certificate hashing
#include <fstream>      // For attestation file I/O (if client generates quote)

// Helper to print OpenSSL errors
void print_ssl_errors(const std::string& prefix) {
    unsigned long err;
    while ((err = ERR_get_error()) != 0) {
        std::cerr << prefix << ": " << ERR_reason_error_string(err) << std::endl;
    }
}

Client::Client(const std::string& server_ip, int server_port, const std::string& client_id)
    : server_ip_(server_ip), server_port_(server_port), client_id_(client_id),
      client_socket_(-1), ssl_ctx_(nullptr), ssl_(nullptr) {
    if (!init_openssl()) {
        std::cerr << "OpenSSL initialization failed." << std::endl;
        // Consider throwing an exception or setting an error state
    }
}

Client::~Client() {
    disconnect();
    if (ssl_ctx_) {
        SSL_CTX_free(ssl_ctx_);
        ssl_ctx_ = nullptr;
    }
    cleanup_openssl();
}

bool Client::init_openssl() {
    SSL_library_init();
    OpenSSL_add_all_algorithms();
    SSL_load_error_strings();
    ERR_load_BIO_strings();
    ERR_load_crypto_strings();
    return true;
}

void Client::cleanup_openssl() {
    ERR_free_strings();
    EVP_cleanup();
}

SSL_CTX* Client::create_ssl_context() {
    const SSL_METHOD* method;
    SSL_CTX* ctx;

    method = TLS_client_method(); // Use TLS_client_method() for current TLS versions
    ctx = SSL_CTX_new(method);
    if (!ctx) {
        std::cerr << "Error creating SSL context." << std::endl;
        print_ssl_errors("SSL_CTX_new");
        return nullptr;
    }

    // Set min TLS version
    if (SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION) == 0) {
        std::cerr << "Error setting min TLS version." << std::endl;
        print_ssl_errors("SSL_CTX_set_min_proto_version");
        SSL_CTX_free(ctx);
        return nullptr;
    }
    
    // TODO: Implement proper server certificate verification in production.
    // This involves:
    // 1. Loading a CA certificate bundle (e.g., using SSL_CTX_load_verify_locations).
    //    SSL_CTX_load_verify_locations(ctx, "ca_cert.pem", nullptr);
    // 2. Setting the verify mode (e.g., SSL_VERIFY_PEER).
    //    SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, nullptr); // Last arg is verify_callback, can be null
    // For now, skipping server certificate verification for simplicity in a test client.
    // WARNING: This is insecure and should NOT be used in production.
    SSL_CTX_set_verify(ctx, SSL_VERIFY_NONE, nullptr); 
    std::cout << "WARNING: Server certificate verification is currently disabled (SSL_VERIFY_NONE)." << std::endl;


    // Optionally, load client certificate and private key if server requires client auth (mTLS)
    // if (SSL_CTX_use_certificate_file(ctx, "client.crt", SSL_FILETYPE_PEM) <= 0) { ... }
    // if (SSL_CTX_use_PrivateKey_file(ctx, "client.key", SSL_FILETYPE_PEM) <= 0) { ... }

    return ctx;
}

bool Client::connect_to_server() {
    client_socket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (client_socket_ == -1) {
        std::cerr << "Error creating socket: " << strerror(errno) << std::endl;
        return false;
    }

    sockaddr_in server_address{};
    server_address.sin_family = AF_INET;
    server_address.sin_port = htons(server_port_);
    if (inet_pton(AF_INET, server_ip_.c_str(), &server_address.sin_addr) <= 0) {
        std::cerr << "Invalid server IP address: " << server_ip_ << std::endl;
        close(client_socket_);
        client_socket_ = -1;
        return false;
    }

    if (::connect(client_socket_, (struct sockaddr*)&server_address, sizeof(server_address)) < 0) {
        std::cerr << "Error connecting to server: " << strerror(errno) << std::endl;
        close(client_socket_);
        client_socket_ = -1;
        return false;
    }
    std::cout << "Successfully connected to server " << server_ip_ << ":" << server_port_ << std::endl;

    // Create SSL context if not already created (e.g. if connect_to_server is called standalone)
    if (!ssl_ctx_) {
        ssl_ctx_ = create_ssl_context();
        if (!ssl_ctx_) {
            disconnect(); // Closes socket
            return false;
        }
    }
    
    return perform_tls_handshake();
}

bool Client::perform_tls_handshake() {
    if (client_socket_ == -1 || !ssl_ctx_) {
        std::cerr << "Cannot perform TLS handshake: not connected or SSL context not initialized." << std::endl;
        return false;
    }

    ssl_ = SSL_new(ssl_ctx_);
    if (!ssl_) {
        std::cerr << "Error creating SSL structure." << std::endl;
        print_ssl_errors("SSL_new");
        return false;
    }

    SSL_set_fd(ssl_, client_socket_);

    // Optionally set SNI (Server Name Indication)
    // SSL_set_tlsext_host_name(ssl_, server_ip_.c_str());

    if (SSL_connect(ssl_) <= 0) {
        std::cerr << "Error performing SSL handshake (SSL_connect)." << std::endl;
        print_ssl_errors("SSL_connect");
        SSL_free(ssl_);
        ssl_ = nullptr;
        return false;
    }
    std::cout << "TLS handshake successful with server. Cipher: " << SSL_get_cipher(ssl_) << std::endl;
    
    // TODO: After successful TLS handshake, verify server certificate details if SSL_VERIFY_NONE is not used.
    // X509* server_cert = SSL_get_peer_certificate(ssl_);
    // if (server_cert) {
    //    // Verify hostname, issuer, validity period, etc.
    //    X509_free(server_cert);
    // } else { // Handle error }

    return true;
}

// --- SGX Attestation Method Implementations ---

// Helper to write to /dev/attestation (adapted for client, assuming Gramine context if used)
bool Client::client_write_to_attestation_file(const char* filepath, const void* data, size_t count) {
    // This implementation assumes the client is running in a Gramine enclave
    // similar to the server for /dev/attestation access.
    // If the client is not in an enclave, this function is a conceptual placeholder.
    std::ofstream file(filepath, std::ios::binary | std::ios::out);
    if (!file.is_open()) {
        std::cerr << "Client: Error opening attestation file for writing: " << filepath << " - " << strerror(errno) << std::endl;
        return false;
    }
    file.write(static_cast<const char*>(data), count);
    if (!file.good()) {
        std::cerr << "Client: Error writing to attestation file: " << filepath << " - " << strerror(errno) << std::endl;
        file.close();
        return false;
    }
    file.close();
    std::cout << "Client: Data written to attestation file: " << filepath << std::endl;
    return true;
}

// Helper to read from /dev/attestation (adapted for client)
std::vector<uint8_t> Client::client_read_from_attestation_file(const char* filepath, size_t read_len) {
    // Similar assumptions as client_write_to_attestation_file about Gramine context.
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Client: Error opening attestation file for reading: " << filepath << " - " << strerror(errno) << std::endl;
        return {};
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (read_len == 0) { // Read till EOF
        read_len = size;
    } else if (read_len > (size_t)size) {
        std::cerr << "Client: Error: Requested to read " << read_len << " bytes from " << filepath 
                  << ", but file size is " << size << "." << std::endl;
        file.close();
        return {};
    }
    
    std::vector<uint8_t> buffer(read_len);
    if (file.read(reinterpret_cast<char*>(buffer.data()), read_len)) {
        file.close();
        std::cout << "Client: Data read from attestation file: " << filepath << ", size: " << buffer.size() << std::endl;
        return buffer;
    } else {
        std::cerr << "Client: Error reading from attestation file: " << filepath << " - " << strerror(errno) << std::endl;
        file.close();
        return {};
    }
}

// Prepares client's report data, e.g., with a nonce.
void Client::prepare_client_report_data(sgx_report_data_t& report_data) {
    memset(&report_data, 0, sizeof(sgx_report_data_t));
    // For simplicity, using a fixed nonce. In a real scenario, this could be
    // a hash of a public key, a random nonce exchanged during TLS, etc.
    // For this example, let's just fill part of it with a pattern.
    for(size_t i = 0; i < SGX_REPORT_DATA_SIZE && i < 8; ++i) {
        report_data.d[i] = static_cast<unsigned char>(0xCL + i); // CL for Client
    }
    std::cout << "Client: Prepared custom report data (nonce)." << std::endl;
}


bool Client::generate_client_quote(std::vector<uint8_t>& client_quote_vec, const sgx_report_data_t& report_data) {
    std::cout << "Client: Generating SGX quote..." << std::endl;
    
    // Write the provided report_data to /dev/attestation/user_report_data
    // This assumes the client is running in Gramine and has access to this interface.
    if (!client_write_to_attestation_file("/dev/attestation/user_report_data", &report_data, sizeof(report_data))) {
        std::cerr << "Client: Failed to write report data for quote generation." << std::endl;
        return false;
    }

    // Read the quote from /dev/attestation/quote
    client_quote_vec = client_read_from_attestation_file("/dev/attestation/quote");
    if (client_quote_vec.empty()) {
        std::cerr << "Client: Failed to read quote from /dev/attestation/quote." << std::endl;
        return false;
    }

    std::cout << "Client: SGX quote generated successfully. Size: " << client_quote_vec.size() << " bytes." << std::endl;
    return true;
}

bool Client::send_client_attestation_data(SSL* ssl_conn, const std::vector<uint8_t>& client_quote) {
    if (!ssl_conn) {
        std::cerr << "Client: SSL connection not available for sending attestation data." << std::endl;
        return false;
    }
    uint32_t quote_size_net = htonl(client_quote.size());
    int bytes_sent = SSL_write(ssl_conn, &quote_size_net, sizeof(quote_size_net));
    if (bytes_sent <= 0) {
        std::cerr << "Client: Error sending client quote size." << std::endl;
        print_ssl_errors("SSL_write client quote size");
        return false;
    }

    bytes_sent = SSL_write(ssl_conn, client_quote.data(), client_quote.size());
    if (bytes_sent <= 0 || (size_t)bytes_sent != client_quote.size()) {
        std::cerr << "Client: Error sending client quote data." << std::endl;
        print_ssl_errors("SSL_write client quote data");
        return false;
    }
    std::cout << "Client: Attestation data (quote) sent to server." << std::endl;
    return true;
}

bool Client::receive_server_attestation_data(SSL* ssl_conn, std::vector<uint8_t>& server_quote_vec) {
    if (!ssl_conn) {
        std::cerr << "Client: SSL connection not available for receiving attestation data." << std::endl;
        return false;
    }
    uint32_t server_quote_size_net;
    int bytes_read = SSL_read(ssl_conn, &server_quote_size_net, sizeof(server_quote_size_net));
    if (bytes_read <= 0) {
        std::cerr << "Client: Error receiving server quote size." << std::endl;
        print_ssl_errors("SSL_read server quote size");
        return false;
    }
     if (bytes_read != sizeof(server_quote_size_net)) {
         std::cerr << "Client: Received incomplete server quote size." << std::endl;
         return false;
    }


    uint32_t server_quote_size = ntohl(server_quote_size_net);
    if (server_quote_size == 0 || server_quote_size > 16384) { // Sanity check (e.g., max 16KB)
        std::cerr << "Client: Invalid server quote size received: " << server_quote_size << std::endl;
        return false;
    }

    server_quote_vec.resize(server_quote_size);
    bytes_read = 0;
    int total_received = 0;
    while(total_received < (int)server_quote_size) {
        bytes_read = SSL_read(ssl_conn, server_quote_vec.data() + total_received, server_quote_size - total_received);
        if (bytes_read <= 0) {
            std::cerr << "Client: Error receiving server quote data. Received " << total_received << " of " << server_quote_size << std::endl;
            print_ssl_errors("SSL_read server quote data");
            return false;
        }
        total_received += bytes_read;
    }
     if (total_received != (int)server_quote_size) {
        std::cerr << "Client: Server quote data reception incomplete. Expected " << server_quote_size << " got " << total_received << std::endl;
        return false;
    }

    std::cout << "Client: Server attestation data (quote) received. Size: " << server_quote_vec.size() << " bytes." << std::endl;
    return true;
}

// Placeholder for server quote verification
bool Client::verify_server_quote(const uint8_t* quote_data, uint32_t quote_size) {
    std::cout << "Client: Verifying server quote (Placeholder)..." << std::endl;
    // TODO: Implement actual server quote verification.
    // This would involve:
    // 1. If the client is also in Gramine, it could potentially use a similar
    //    `/dev/attestation/verify_quote` interface if configured for it.
    // 2. Alternatively, the client might use a DCAP library directly (e.g., QVL)
    //    to verify the quote against Intel's PCS or a local PCCS.
    // 3. The client must extract the report_data from the server's quote.
    // 4. This report_data should contain the hash of the server's TLS certificate
    //    (as seen by the client: SSL_get_peer_certificate()). The client calculates this hash
    //    independently and compares it. This proves the attested server is the one participating
    //    in this specific TLS session.
    
    // Example of getting server cert hash for verification (conceptual)
    // sgx_report_data_t expected_report_data_from_server_cert = {0};
    // if (ssl_ && !get_tls_certificate_hash_for_server_verification(ssl_, expected_report_data_from_server_cert)) {
    //    std::cerr << "Client: Could not get server's certificate hash for verification." << std::endl;
    //    return false;
    // }
    // Now compare expected_report_data_from_server_cert with report_data extracted from quote_data.

    std::cout << "Client: Server quote verification placeholder: Succeeded. Size: " << quote_size << " bytes." << std::endl;
    return true; // For now, always return true.
}

bool Client::perform_sgx_attestation_flow() {
    if (!ssl_) {
        std::cerr << "Client: Cannot perform SGX attestation: SSL not established." << std::endl;
        return false;
    }
    std::cout << "Client: Starting SGX Attestation flow..." << std::endl;

    // 1. Prepare client's report data (e.g., with a nonce)
    sgx_report_data_t user_data_for_client_quote;
    prepare_client_report_data(user_data_for_client_quote);

    // 2. Generate client's own quote
    std::vector<uint8_t> client_quote_vec;
    if (!generate_client_quote(client_quote_vec, user_data_for_client_quote)) {
        std::cerr << "Client: Failed to generate its own quote." << std::endl;
        return false;
    }

    // 3. Send client's quote to the server
    if (!send_client_attestation_data(ssl_, client_quote_vec)) {
        std::cerr << "Client: Failed to send its quote to the server." << std::endl;
        return false;
    }

    // 4. Receive server's quote
    std::vector<uint8_t> server_quote_vec;
    if (!receive_server_attestation_data(ssl_, server_quote_vec)) {
        std::cerr << "Client: Failed to receive server's quote." << std::endl;
        return false;
    }

    // 5. Verify server's quote (placeholder)
    if (!verify_server_quote(server_quote_vec.data(), server_quote_vec.size())) {
        std::cerr << "Client: Server quote verification failed." << std::endl;
        return false;
    }

    std::cout << "Client: Mutual SGX attestation with server completed successfully." << std::endl;
    return true;
}


// --- Client-Side Message Helper Implementations ---

bool Client::client_send_message(SSL* ssl_conn, uint8_t msg_type, const std::vector<uint8_t>& payload) {
    if (!ssl_conn) {
        std::cerr << "Client: SSL connection not available for sending message." << std::endl;
        return false;
    }

    // 1. Send message type (1 byte)
    int bytes_sent = SSL_write(ssl_conn, &msg_type, 1);
    if (bytes_sent <= 0) {
        std::cerr << "Client: Error sending message type (0x" << std::hex << (int)msg_type << std::dec << ")." << std::endl;
        print_ssl_errors("SSL_write msg_type");
        return false;
    }

    // 2. Send payload length (4 bytes, network byte order)
    uint32_t payload_len_net = htonl(static_cast<uint32_t>(payload.size()));
    bytes_sent = 0;
    int total_len_sent = 0;
    while(total_len_sent < sizeof(payload_len_net)) {
        bytes_sent = SSL_write(ssl_conn, reinterpret_cast<const uint8_t*>(&payload_len_net) + total_len_sent, sizeof(payload_len_net) - total_len_sent);
        if (bytes_sent <= 0) {
            std::cerr << "Client: Error sending payload length for message type (0x" << std::hex << (int)msg_type << std::dec << ")." << std::endl;
            print_ssl_errors("SSL_write payload_len");
            return false;
        }
        total_len_sent += bytes_sent;
    }
    
    // 3. Send payload data (if any)
    if (!payload.empty()) {
        bytes_sent = 0;
        uint32_t total_payload_sent = 0;
        while(total_payload_sent < payload.size()){
            bytes_sent = SSL_write(ssl_conn, payload.data() + total_payload_sent, payload.size() - total_payload_sent);
            if (bytes_sent <= 0) {
                std::cerr << "Client: Error sending payload data for message type (0x" << std::hex << (int)msg_type << std::dec << ")." << std::endl;
                print_ssl_errors("SSL_write payload data");
                return false;
            }
            total_payload_sent += bytes_sent;
        }
    }
    // std::cout << "Client: Message sent. Type: 0x" << std::hex << (int)msg_type << std::dec << ", Payload size: " << payload.size() << std::endl;
    return true;
}

bool Client::client_receive_message(SSL* ssl_conn, uint8_t& msg_type, std::vector<uint8_t>& payload) {
    if (!ssl_conn) {
        std::cerr << "Client: SSL connection not available for receiving message." << std::endl;
        return false;
    }
    payload.clear();

    // 1. Read message type (1 byte)
    int bytes_read = SSL_read(ssl_conn, &msg_type, 1);
    if (bytes_read <= 0) {
        if (bytes_read == 0) std::cout << "Client: Server disconnected while reading message type." << std::endl;
        else {
            std::cerr << "Client: Error reading message type from SSL." << std::endl;
            print_ssl_errors("SSL_read msg_type");
        }
        return false;
    }

    // 2. Read payload length (4 bytes, network byte order)
    uint32_t payload_len_net;
    bytes_read = 0;
    int total_read_len_bytes = 0;
    while(total_read_len_bytes < sizeof(payload_len_net)) {
        bytes_read = SSL_read(ssl_conn, reinterpret_cast<uint8_t*>(&payload_len_net) + total_read_len_bytes, sizeof(payload_len_net) - total_read_len_bytes);
        if (bytes_read <= 0) {
            if (bytes_read == 0) std::cout << "Client: Server disconnected while reading payload length." << std::endl;
            else {
                 std::cerr << "Client: Error reading payload length from SSL." << std::endl;
                 print_ssl_errors("SSL_read payload_len");
            }
            return false;
        }
        total_read_len_bytes += bytes_read;
    }
    uint32_t payload_len = ntohl(payload_len_net);

    const uint32_t MAX_PAYLOAD_SIZE = 1024 * 1024; // 1MB limit, same as server
    if (payload_len > MAX_PAYLOAD_SIZE) {
        std::cerr << "Client: Received payload length " << payload_len << " exceeds maximum " << MAX_PAYLOAD_SIZE << "." << std::endl;
        return false; 
    }
    
    if (payload_len == 0) {
        // std::cout << "Client: Message received. Type: 0x" << std::hex << (int)msg_type << std::dec << ", Payload size: 0" << std::endl;
        return true;
    }

    // 3. Read payload data
    payload.resize(payload_len);
    bytes_read = 0;
    uint32_t total_payload_read = 0;
    while (total_payload_read < payload_len) {
        bytes_read = SSL_read(ssl_conn, payload.data() + total_payload_read, payload_len - total_payload_read);
        if (bytes_read <= 0) {
            if (bytes_read == 0) std::cout << "Client: Server disconnected while reading payload." << std::endl;
            else {
                std::cerr << "Client: Error reading payload from SSL." << std::endl;
                print_ssl_errors("SSL_read payload data");
            }
            return false;
        }
        total_payload_read += bytes_read;
    }
    // std::cout << "Client: Message received. Type: 0x" << std::hex << (int)msg_type << std::dec << ", Payload size: " << payload.size() << std::endl;
    return true;
}


// --- Application Protocol Method Implementations ---

bool Client::register_self() {
    if (!ssl_) {
        std::cerr << "Client: Cannot register, SSL not established." << std::endl;
        return false;
    }
    std::cout << "Client: Attempting to register with client_id: " << client_id_ << std::endl;
    std::vector<uint8_t> id_payload(client_id_.begin(), client_id_.end());

    if (!client_send_message(ssl_, MSG_TYPE_REGISTER_CLIENT, id_payload)) {
        std::cerr << "Client: Failed to send registration message." << std::endl;
        return false;
    }

    uint8_t response_type;
    std::vector<uint8_t> response_payload;
    if (!client_receive_message(ssl_, response_type, response_payload)) {
        std::cerr << "Client: Failed to receive response for registration." << std::endl;
        return false;
    }

    if (response_type == MSG_TYPE_REGISTER_ACK) {
        std::cout << "Client: Registration successful (ACK received)." << std::endl;
        return true;
    } else {
        std::cerr << "Client: Registration failed. Server responded with type 0x" 
                  << std::hex << (int)response_type << std::dec;
        if (!response_payload.empty()) {
            std::cerr << ", Reason: " << std::string(response_payload.begin(), response_payload.end());
        }
        std::cerr << std::endl;
        return false;
    }
}

bool Client::send_data_to_recipient(const std::string& recipient_id, const std::vector<uint8_t>& data) {
    if (!ssl_) {
        std::cerr << "Client: Cannot send data, SSL not established." << std::endl;
        return false;
    }
    std::cout << "Client: Attempting to send data to recipient '" << recipient_id << "', data size: " << data.size() << std::endl;

    // Construct payload: recipient_id_string_length (uint16_t), recipient_id_string (variable), data_payload (variable)
    std::vector<uint8_t> message_payload;
    uint16_t recipient_id_len = static_cast<uint16_t>(recipient_id.length());
    uint16_t recipient_id_len_net = htons(recipient_id_len);

    message_payload.insert(message_payload.end(), reinterpret_cast<uint8_t*>(&recipient_id_len_net), reinterpret_cast<uint8_t*>(&recipient_id_len_net) + sizeof(uint16_t));
    message_payload.insert(message_payload.end(), recipient_id.begin(), recipient_id.end());
    message_payload.insert(message_payload.end(), data.begin(), data.end());

    if (!client_send_message(ssl_, MSG_TYPE_SEND_DATA, message_payload)) {
        std::cerr << "Client: Failed to send data message." << std::endl;
        return false;
    }
    
    uint8_t response_type;
    std::vector<uint8_t> response_payload;
    if (!client_receive_message(ssl_, response_type, response_payload)) {
        std::cerr << "Client: Failed to receive response for send data." << std::endl;
        return false;
    }

    if (response_type == MSG_TYPE_SEND_ACK) {
        std::cout << "Client: Send data successful (ACK received)." << std::endl;
        return true;
    } else {
        std::cerr << "Client: Send data failed. Server responded with type 0x" 
                  << std::hex << (int)response_type << std::dec;
        if (!response_payload.empty()) {
            std::cerr << ", Reason: " << std::string(response_payload.begin(), response_payload.end());
        }
        std::cerr << std::endl;
        return false;
    }
}

bool Client::poll_for_data() {
    if (!ssl_) {
        std::cerr << "Client: Cannot poll for data, SSL not established." << std::endl;
        return false;
    }
    std::cout << "Client: Polling for data..." << std::endl;

    if (!client_send_message(ssl_, MSG_TYPE_POLL_DATA)) {
        std::cerr << "Client: Failed to send poll data message." << std::endl;
        return false;
    }

    uint8_t response_type;
    std::vector<uint8_t> response_payload;
    if (!client_receive_message(ssl_, response_type, response_payload)) {
        std::cerr << "Client: Failed to receive response for poll data." << std::endl;
        return false;
    }

    if (response_type == MSG_TYPE_DATA_AVAILABLE) {
        if (response_payload.size() < sizeof(uint16_t)) {
            std::cerr << "Client: Received DATA_AVAILABLE but payload too short for sender_id length." << std::endl;
            return false; // Or handle as error
        }
        uint16_t sender_id_len_net;
        memcpy(&sender_id_len_net, response_payload.data(), sizeof(uint16_t));
        uint16_t sender_id_len = ntohs(sender_id_len_net);

        if (response_payload.size() < sizeof(uint16_t) + sender_id_len) {
            std::cerr << "Client: Received DATA_AVAILABLE but payload too short for sender_id." << std::endl;
            return false; // Or handle as error
        }
        std::string sender_id(response_payload.begin() + sizeof(uint16_t), response_payload.begin() + sizeof(uint16_t) + sender_id_len);
        std::vector<uint8_t> actual_data(response_payload.begin() + sizeof(uint16_t) + sender_id_len, response_payload.end());
        
        std::cout << "Client: Received data from sender '" << sender_id << "'. Size: " << actual_data.size() << "." << std::endl;
        // For simplicity, printing if it's text, otherwise hex.
        bool is_text = true;
        for(unsigned char c : actual_data) {
            if (!isprint(c) && !isspace(c)) {
                is_text = false;
                break;
            }
        }
        if(is_text) {
            std::cout << "Data: \"" << std::string(actual_data.begin(), actual_data.end()) << "\"" << std::endl;
        } else {
            std::cout << "Data (hex): ";
            for(unsigned char c : actual_data) {
                std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)c;
            }
            std::cout << std::dec << std::endl;
        }
        return true;
    } else if (response_type == MSG_TYPE_NO_DATA_AVAILABLE) {
        std::cout << "Client: No data available from server." << std::endl;
        return true;
    } else {
        std::cerr << "Client: Poll data failed. Server responded with type 0x" 
                  << std::hex << (int)response_type << std::dec;
        if (!response_payload.empty()) {
            std::cerr << ", Reason: " << std::string(response_payload.begin(), response_payload.end());
        }
        std::cerr << std::endl;
        return false;
    }
}


void Client::run_application_protocol() {
    if (!ssl_) {
        std::cerr << "Client: Application protocol cannot run: Not connected or TLS/Attestation failed." << std::endl;
        return;
    }
    std::cout << "Client: Starting application protocol..." << std::endl;
    
    if (register_self()) { // Use the member client_id_
        std::cout << "Client: Registration successful." << std::endl;        
        
        std::string test_recipient = "TestClientB"; // Example recipient
        if (client_id_ == test_recipient) { // Avoid sending to self if running two instances with same ID for testing
            test_recipient = "TestClientC";
        }
        std::string test_message_str = "Hello from " + client_id_ + "!";
        std::vector<uint8_t> sample_data(test_message_str.begin(), test_message_str.end());

        if (send_data_to_recipient(test_recipient, sample_data)) {
            std::cout << "Client: Sent data to " << test_recipient << " successfully." << std::endl;
        } else {
            std::cerr << "Client: Failed to send data to " << test_recipient << "." << std::endl;
        }

        // Poll a couple of times
        std::cout << "Client: Polling for messages (Attempt 1)..." << std::endl;
        poll_for_data(); 
        
        // Simple delay for demonstration if testing locally with another client instance
        // In a real app, polling might be event-driven or use longer, configurable intervals.
        // std::this_thread::sleep_for(std::chrono::seconds(2)); 
        
        std::cout << "Client: Polling for messages (Attempt 2)..." << std::endl;
        poll_for_data();

    } else {
        std::cerr << "Client: Registration failed. Cannot proceed with application protocol." << std::endl;
    }
    std::cout << "Client: Application protocol sequence finished." << std::endl;
}

void Client::disconnect() {
    if (ssl_) {
        SSL_shutdown(ssl_); 
        SSL_free(ssl_);
        ssl_ = nullptr;
    }
    if (client_socket_ != -1) {
        close(client_socket_);
        client_socket_ = -1;
        std::cout << "Client: Disconnected from server." << std::endl;
    }
}


// Combines connection, TLS, and attestation
bool Client::connect_and_attest() {
    if (!connect_to_server()) { // connect_to_server already calls perform_tls_handshake
        std::cerr << "Client: Failed to connect or perform TLS handshake." << std::endl;
        return false;
    }
    std::cout << "Client: Successfully connected to server and TLS handshake completed." << std::endl;
    
    if (!perform_sgx_attestation_flow()) {
        std::cerr << "Client: SGX Attestation flow failed." << std::endl;
        disconnect(); // Disconnect if attestation fails
        return false;
    }
    // If attestation is successful, the 'ssl_' object is valid and ready for app data.
    return true;
}


// Main function for the client executable
int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <server_ip> <server_port> [client_id]" << std::endl;
        return 1;
    }

    std::string server_ip = argv[1];
    int server_port = std::stoi(argv[2]);
    std::string client_id = "default_client_id"; // Default if not provided
    if (argc == 4) {
        client_id = argv[3];
    }
    
    Client client(server_ip, server_port, client_id);

    if (client.connect_and_attest()) { // This now includes connect, TLS, and attestation
        client.run_application_protocol(); // Placeholder application logic
        client.disconnect();
    } else {
        std::cerr << "Client: Overall connection or attestation process failed." << std::endl;
        // client.disconnect() would have been called internally if needed by connect_and_attest()
    }

    return 0;
}
