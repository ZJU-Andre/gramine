#include "server.h"
#include <iostream>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <signal.h> // Required for signal handling
#include <openssl/ssl.h>
#include <openssl/err.h>

// Global atomic flag to signal server shutdown
std::atomic<bool> g_shutdown_flag(false);

// Signal handler function
void signal_handler(int signum) {
    std::cout << "Signal " << signum << " received, shutting down server." << std::endl;
    g_shutdown_flag = true;
}

void Server::init_openssl() {
    SSL_library_init();
    OpenSSL_add_all_algorithms();
    SSL_load_error_strings();
    ERR_load_BIO_strings(); // Load BIO error strings
    ERR_load_crypto_strings(); // Load crypto error strings
}

void Server::cleanup_openssl() {
    ERR_free_strings();
    EVP_cleanup();
}

SSL_CTX* Server::create_ssl_context() {
    const SSL_METHOD* method;
    SSL_CTX* ctx;

    method = TLS_server_method(); // Use TLS_server_method() for current TLS versions

    ctx = SSL_CTX_new(method);
    if (!ctx) {
        std::cerr << "Error creating SSL context." << std::endl;
        ERR_print_errors_fp(stderr);
        return nullptr;
    }

    // Set min/max TLS versions
    if (SSL_CTX_set_min_proto_version(ctx, TLS1_2_VERSION) == 0) {
        std::cerr << "Error setting min TLS version." << std::endl;
        ERR_print_errors_fp(stderr);
        SSL_CTX_free(ctx);
        return nullptr;
    }
    // SSL_CTX_set_max_proto_version can also be used if needed.

    return ctx;
}

void Server::configure_ssl_context(SSL_CTX* ctx) {
    // Note: server.crt and server.key must be generated, e.g., using:
    // openssl req -x509 -newkey rsa:4096 -keyout server.key -out server.crt -days 365 -nodes

    if (SSL_CTX_use_certificate_file(ctx, cert_path_, SSL_FILETYPE_PEM) <= 0) {
        std::cerr << "Error loading certificate file: " << cert_path_ << std::endl;
        ERR_print_errors_fp(stderr);
        return; // Or throw exception
    }

    if (SSL_CTX_use_PrivateKey_file(ctx, key_path_, SSL_FILETYPE_PEM) <= 0) {
        std::cerr << "Error loading private key file: " << key_path_ << std::endl;
        ERR_print_errors_fp(stderr);
        return; // Or throw exception
    }

    if (!SSL_CTX_check_private_key(ctx)) {
        std::cerr << "Private key does not match the public certificate." << std::endl;
        ERR_print_errors_fp(stderr);
        return; // Or throw exception
    }
    std::cout << "Certificate and private key loaded successfully." << std::endl;
}

Server::Server(int port, const char* cert_path, const char* key_path)
    : port_(port), cert_path_(cert_path), key_path_(key_path),
      server_socket_(-1), running_(false), ssl_ctx_(nullptr) {}

Server::~Server() {
    if (running_) {
        stop();
    }
    if (server_socket_ != -1) {
        close(server_socket_);
    }
    if (ssl_ctx_) {
        SSL_CTX_free(ssl_ctx_);
        ssl_ctx_ = nullptr;
    }
    cleanup_openssl(); // Cleanup OpenSSL resources
}

void Server::start() {
    init_openssl();
    ssl_ctx_ = create_ssl_context();
    if (!ssl_ctx_) {
        return; // Error already printed
    }
    configure_ssl_context(ssl_ctx_);
    // Check if configure_ssl_context failed (e.g., by checking if a required cert/key is loaded)
    // For simplicity, we assume configure_ssl_context prints errors and we proceed,
    // but a more robust implementation might set a flag in Server or throw.
    if (!SSL_CTX_get_ex_data(ssl_ctx_, 0)) { // A simple check, not foolproof
        // This check is not standard. A better way is to ensure configure_ssl_context sets a status.
        // Or check return values of SSL_CTX_use_certificate_file & SSL_CTX_use_PrivateKey_file directly.
        // For now, we rely on the error messages from configure_ssl_context.
        // If cert/key loading failed, subsequent SSL_accept will fail.
    }


    // Register signal handlers for SIGINT and SIGTERM
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    server_socket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket_ == -1) {
        std::cerr << "Error creating socket." << std::endl;
        // Consider cleaning up SSL_CTX if socket creation fails early
        // SSL_CTX_free(ssl_ctx_); ssl_ctx_ = nullptr; cleanup_openssl();
        return;
    }

    sockaddr_in server_address{};
    server_address.sin_family = AF_INET;
    server_address.sin_addr.s_addr = INADDR_ANY;
    server_address.sin_port = htons(port_);

    if (bind(server_socket_, (struct sockaddr*)&server_address, sizeof(server_address)) < 0) {
        std::cerr << "Error binding socket." << std::endl;
        close(server_socket_);
        server_socket_ = -1;
        // SSL_CTX_free(ssl_ctx_); ssl_ctx_ = nullptr; cleanup_openssl();
        return;
    }

    if (listen(server_socket_, 10) < 0) { // Increased backlog slightly
        std::cerr << "Error listening on socket." << std::endl;
        close(server_socket_);
        server_socket_ = -1;
        // SSL_CTX_free(ssl_ctx_); ssl_ctx_ = nullptr; cleanup_openssl();
        return;
    }

    running_ = true;
    accept_thread_ = std::thread(&Server::accept_connections, this);
    std::cout << "Server started on port " << port_ << ", waiting for connections..." << std::endl;
}

void Server::stop() {
    running_ = false;
    g_shutdown_flag = true; 

    if (server_socket_ != -1) {
        // Closing the socket abruptly can cause issues with ongoing SSL handshakes
        // or data transmission. A more graceful shutdown might involve signaling
        // active client handlers to terminate.
        // For now, shutdown read to unblock accept().
        shutdown(server_socket_, SHUT_RD);
    }

    if (accept_thread_.joinable()) {
        accept_thread_.join();
    }

    if (server_socket_ != -1) {
        close(server_socket_);
        server_socket_ = -1;
    }
    
    // SSL_CTX_free is handled in the destructor
    // cleanup_openssl(); // Also handled in destructor

    std::cout << "Server stopped." << std::endl;
}

void Server::accept_connections() {
    while (running_ && !g_shutdown_flag) {
        sockaddr_in client_address{};
        socklen_t client_len = sizeof(client_address);
        int client_socket = accept(server_socket_, (struct sockaddr*)&client_address, &client_len);

        if (g_shutdown_flag || !running_) {
            if (client_socket >= 0) close(client_socket);
            break;
        }

        if (client_socket < 0) {
            if (errno == EINTR && g_shutdown_flag) {
                std::cout << "Accept interrupted by signal, shutting down." << std::endl;
                break;
            }
            if (running_ && server_socket_ != -1) {
                 std::cerr << "Error accepting client connection: " << strerror(errno) << std::endl;
            }
            continue;
        }

        std::cout << "Client connected (raw socket: " << client_socket << "). Performing TLS handshake..." << std::endl;

        SSL* ssl = SSL_new(ssl_ctx_);
        if (!ssl) {
            std::cerr << "Error creating SSL structure." << std::endl;
            ERR_print_errors_fp(stderr);
            close(client_socket);
            continue;
        }
        SSL_set_fd(ssl, client_socket);

        if (SSL_accept(ssl) <= 0) {
            std::cerr << "Error performing SSL handshake. Socket: " << client_socket << std::endl;
            ERR_print_errors_fp(stderr);
            SSL_free(ssl);
            close(client_socket);
            continue;
        }
        std::cout << "TLS handshake successful for client (socket: " << client_socket << ")" << std::endl;

        // TODO: Implement SGX Attestation here (after TLS handshake)
        // For example:
        // sgx_report_data_t report_data;
        // generate_attestation_report(ssl, &report_data); // Pass SSL object
        // verify_attestation_report(ssl, &report_data);

        // Pass SSL* to handle_client
        std::thread client_thread(&Server::handle_client_tls, this, ssl); // Renamed to handle_client_tls
        client_thread.detach();
    }
    if (running_) {
        running_ = false;
    }
    std::cout << "Accept connections loop finished." << std::endl;
}

#include <fstream> // For file I/O
#include <arpa/inet.h> // For htonl/ntohl

// --- Start of SGX Attestation Helper Implementations ---

// Helper function to write data to a /dev/attestation/ file
bool Server::write_to_attestation_file(const char* filepath, const void* data, size_t count) {
    std::ofstream file(filepath, std::ios::binary | std::ios::out);
    if (!file.is_open()) {
        std::cerr << "Error opening attestation file for writing: " << filepath << " - " << strerror(errno) << std::endl;
        return false;
    }
    file.write(static_cast<const char*>(data), count);
    if (!file.good()) {
        std::cerr << "Error writing to attestation file: " << filepath << " - " << strerror(errno) << std::endl;
        file.close();
        return false;
    }
    file.close();
    return true;
}

// Helper function to read data from a /dev/attestation/ file
// read_len = 0 means read until EOF. Otherwise, reads exactly read_len bytes.
std::vector<uint8_t> Server::read_from_attestation_file(const char* filepath, size_t read_len) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate); // Open at the end to get size
    if (!file.is_open()) {
        std::cerr << "Error opening attestation file for reading: " << filepath << " - " << strerror(errno) << std::endl;
        return {};
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (read_len == 0) { // Read till EOF
        read_len = size;
    } else if (read_len > (size_t)size) { // Requested more than available
        std::cerr << "Error: Requested to read " << read_len << " bytes from " << filepath 
                  << ", but file size is " << size << "." << std::endl;
        file.close();
        return {};
    }
    
    std::vector<uint8_t> buffer(read_len);
    if (file.read(reinterpret_cast<char*>(buffer.data()), read_len)) {
        file.close();
        return buffer;
    } else {
        std::cerr << "Error reading from attestation file: " << filepath << " - " << strerror(errno) << std::endl;
        file.close();
        return {};
    }
}


// Hashes the server's TLS certificate and puts it into report_data
bool Server::get_tls_certificate_hash(SSL* ssl, sgx_report_data_t& report_data) {
    memset(&report_data, 0, sizeof(sgx_report_data_t)); // Clear report data

    X509* cert = SSL_get_peer_certificate(ssl); // In server mode, SSL_get_peer_certificate is for client cert.
                                                // We need the server's own certificate.
    if (!cert) { // Try to get server's own certificate
         SSL_CTX* ctx = SSL_get_SSL_CTX(ssl);
         if (ctx) {
            cert = SSL_CTX_get0_certificate(ctx); // Does not increment ref count
         }
    }
    
    if (!cert) {
        std::cerr << "Error: Could not get server TLS certificate." << std::endl;
        ERR_print_errors_fp(stderr);
        return false;
    }

    unsigned char cert_der[2048]; // Max size for DER encoded cert
    int cert_der_len = i2d_X509(cert, &cert_der); // Convert X509 to DER
    // X509_free(cert); // Do not free if obtained from SSL_CTX_get0_certificate

    if (cert_der_len < 0) {
        std::cerr << "Error converting certificate to DER format." << std::endl;
        ERR_print_errors_fp(stderr);
        return false;
    }

    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, cert_der, cert_der_len);
    SHA256_Final(hash, &sha256);

    // Copy into sgx_report_data_t (typically first 64 bytes are used, SHA256 is 32 bytes)
    // Ensure we don't overflow report_data.d
    size_t copy_len = std::min((size_t)SHA256_DIGEST_LENGTH, (size_t)SGX_REPORT_DATA_SIZE);
    memcpy(report_data.d, hash, copy_len);

    std::cout << "TLS certificate hash generated and placed in report data." << std::endl;
    return true;
}


// Generate server's SGX quote
bool Server::generate_server_quote(SSL* ssl_for_cert_hash, std::vector<uint8_t>& server_quote_vec) {
    sgx_report_data_t report_data_to_embed = {0};

    // Embed hash of server's TLS certificate into the report data
    if (!get_tls_certificate_hash(ssl_for_cert_hash, report_data_to_embed)) {
        std::cerr << "Failed to get TLS certificate hash for server quote." << std::endl;
        return false;
    }

    // Write report data to /dev/attestation/user_report_data
    if (!write_to_attestation_file("/dev/attestation/user_report_data", report_data_to_embed.d, sizeof(report_data_to_embed.d))) {
        std::cerr << "Failed to write report data for server quote generation." << std::endl;
        return false;
    }
    std::cout << "User report data written to /dev/attestation/user_report_data." << std::endl;

    // Read the quote from /dev/attestation/quote
    // The size of the quote can vary. Reading until EOF is common for /dev/attestation/quote.
    server_quote_vec = read_from_attestation_file("/dev/attestation/quote");
    if (server_quote_vec.empty()) {
        std::cerr << "Failed to read server quote from /dev/attestation/quote." << std::endl;
        return false;
    }

    std::cout << "Server quote generated successfully. Size: " << server_quote_vec.size() << " bytes." << std::endl;
    return true;
}

// Send server's attestation data (quote) to the client
bool Server::send_server_attestation_data(SSL* ssl, const std::vector<uint8_t>& server_quote) {
    uint32_t quote_size_net = htonl(server_quote.size()); // Convert to network byte order

    // Send quote size
    int bytes_sent = SSL_write(ssl, &quote_size_net, sizeof(quote_size_net));
    if (bytes_sent <= 0) {
        std::cerr << "Error sending server quote size." << std::endl;
        ERR_print_errors_fp(stderr);
        return false;
    }

    // Send quote data
    bytes_sent = SSL_write(ssl, server_quote.data(), server_quote.size());
    if (bytes_sent <= 0 || (size_t)bytes_sent != server_quote.size()) {
        std::cerr << "Error sending server quote data." << std::endl;
        ERR_print_errors_fp(stderr);
        return false;
    }
    std::cout << "Server quote sent to client." << std::endl;
    return true;
}

// Receive client's attestation data (quote)
bool Server::receive_client_attestation_data(SSL* ssl, std::vector<uint8_t>& client_quote_vec) {
    uint32_t quote_size_net;
    int bytes_received = SSL_read(ssl, &quote_size_net, sizeof(quote_size_net));
    if (bytes_received <= 0) {
        std::cerr << "Error receiving client quote size." << std::endl;
        ERR_print_errors_fp(stderr);
        return false;
    }
    if (bytes_received != sizeof(quote_size_net)) {
         std::cerr << "Received incomplete client quote size." << std::endl;
         return false;
    }

    uint32_t quote_size = ntohl(quote_size_net);
    if (quote_size == 0 || quote_size > 16384) { // Sanity check quote size (e.g., max 16KB)
        std::cerr << "Invalid client quote size received: " << quote_size << std::endl;
        return false;
    }

    client_quote_vec.resize(quote_size);
    bytes_received = 0;
    int total_received = 0;
    // Loop to ensure all quote data is read, as SSL_read might return partial data
    while(total_received < (int)quote_size) {
        bytes_received = SSL_read(ssl, client_quote_vec.data() + total_received, quote_size - total_received);
        if (bytes_received <= 0) {
            std::cerr << "Error receiving client quote data. Received " << total_received << " of " << quote_size << std::endl;
            ERR_print_errors_fp(stderr);
            return false;
        }
        total_received += bytes_received;
    }
    
    if (total_received != (int)quote_size) {
        std::cerr << "Client quote data reception incomplete. Expected " << quote_size << " got " << total_received << std::endl;
        return false;
    }

    std::cout << "Client quote received. Size: " << quote_size << " bytes." << std::endl;
    return true;
}

// Placeholder for client quote verification
bool Server::verify_client_quote(const uint8_t* quote_data, uint32_t quote_size) {
    std::cout << "Verifying client quote (Placeholder)..." << std::endl;
    // TODO: Implement actual client quote verification using Gramine's mechanisms.
    // This would typically involve:
    // 1. (Optional) If client needs server's target_info: Server generates target_info, sends to client.
    //    Client generates quote using this target_info.
    // 2. Client sends its quote to server (this function's input).
    // 3. Server verifies the quote. This might involve:
    //    a. Writing the quote to a file.
    //    b. Providing expected values (e.g., PCRs, MRSIGNER, MRENCLAVE, report data containing client's TLS cert hash)
    //       to a verification oracle or library (e.g. DCAP quote verification library).
    //    c. For DCAP, this might involve interacting with `/dev/attestation/verify_quote` or similar,
    //       or using a library that handles QVL/PCCS interactions.
    //    d. The report data embedded in the client's quote should be checked to ensure it binds
    //       the quote to this specific TLS session (e.g., hash of client's TLS certificate or a nonce).
    std::cout << "Client quote verification placeholder: Succeeded." << std::endl;
    return true; // For now, always return true.
}

// --- End of SGX Attestation Helper Implementations ---

// --- Start of Application-Level Messaging Helper Implementations ---

// Read a message from the client (type, length, payload)
bool Server::read_message(SSL* ssl, uint8_t& msg_type, std::vector<uint8_t>& payload) {
    // 1. Read message type (1 byte)
    int bytes_read = SSL_read(ssl, &msg_type, 1);
    if (bytes_read <= 0) {
        if (bytes_read == 0) std::cout << "Client disconnected while reading message type." << std::endl;
        else {
            std::cerr << "Error reading message type from SSL. SSL_read returned: " << bytes_read << std::endl;
            ERR_print_errors_fp(stderr);
        }
        return false;
    }

    // 2. Read payload length (4 bytes, network byte order)
    uint32_t payload_len_net;
    bytes_read = 0;
    int total_read_len_bytes = 0;
    while(total_read_len_bytes < sizeof(payload_len_net)) {
        bytes_read = SSL_read(ssl, reinterpret_cast<uint8_t*>(&payload_len_net) + total_read_len_bytes, sizeof(payload_len_net) - total_read_len_bytes);
        if (bytes_read <= 0) {
            if (bytes_read == 0) std::cout << "Client disconnected while reading payload length." << std::endl;
            else {
                 std::cerr << "Error reading payload length from SSL. SSL_read returned: " << bytes_read << std::endl;
                 ERR_print_errors_fp(stderr);
            }
            return false;
        }
        total_read_len_bytes += bytes_read;
    }

    uint32_t payload_len = ntohl(payload_len_net);

    // Sanity check payload length (e.g., max 1MB for now)
    const uint32_t MAX_PAYLOAD_SIZE = 1024 * 1024; 
    if (payload_len > MAX_PAYLOAD_SIZE) {
        std::cerr << "Payload length " << payload_len << " exceeds maximum allowed size " << MAX_PAYLOAD_SIZE << "." << std::endl;
        // It's tricky to recover from this cleanly with current SSL state, best to close.
        return false; 
    }
    
    if (payload_len == 0) {
        payload.clear();
        return true;
    }

    // 3. Read payload data
    payload.resize(payload_len);
    bytes_read = 0;
    uint32_t total_payload_read = 0;
    while (total_payload_read < payload_len) {
        bytes_read = SSL_read(ssl, payload.data() + total_payload_read, payload_len - total_payload_read);
        if (bytes_read <= 0) {
            if (bytes_read == 0) std::cout << "Client disconnected while reading payload." << std::endl;
            else {
                std::cerr << "Error reading payload from SSL. SSL_read returned: " << bytes_read << std::endl;
                ERR_print_errors_fp(stderr);
            }
            return false;
        }
        total_payload_read += bytes_read;
    }
    return true;
}

// Send a response to the client (type, length, payload)
bool Server::send_response(SSL* ssl, uint8_t msg_type, const std::vector<uint8_t>& payload) {
    // 1. Send message type (1 byte)
    int bytes_sent = SSL_write(ssl, &msg_type, 1);
    if (bytes_sent <= 0) {
        std::cerr << "Error sending message type via SSL. SSL_write returned: " << bytes_sent << std::endl;
        ERR_print_errors_fp(stderr);
        return false;
    }

    // 2. Send payload length (4 bytes, network byte order)
    uint32_t payload_len_net = htonl(static_cast<uint32_t>(payload.size()));
    bytes_sent = 0;
    int total_len_sent = 0;
    while(total_len_sent < sizeof(payload_len_net)){
        bytes_sent = SSL_write(ssl, reinterpret_cast<const uint8_t*>(&payload_len_net) + total_len_sent, sizeof(payload_len_net) - total_len_sent);
        if (bytes_sent <= 0) {
            std::cerr << "Error sending payload length via SSL. SSL_write returned: " << bytes_sent << std::endl;
            ERR_print_errors_fp(stderr);
            return false;
        }
        total_len_sent += bytes_sent;
    }


    // 3. Send payload data (if any)
    if (!payload.empty()) {
        bytes_sent = 0;
        uint32_t total_payload_sent = 0;
        while(total_payload_sent < payload.size()){
            bytes_sent = SSL_write(ssl, payload.data() + total_payload_sent, payload.size() - total_payload_sent);
            if (bytes_sent <= 0) {
                std::cerr << "Error sending payload data via SSL. SSL_write returned: " << bytes_sent << std::endl;
                ERR_print_errors_fp(stderr);
                return false;
            }
            total_payload_sent += bytes_sent;
        }
    }
    return true;
}

bool Server::send_simple_response(SSL* ssl, uint8_t msg_type) {
    return send_response(ssl, msg_type, {});
}


// --- End of Application-Level Messaging Helper Implementations ---


void Server::handle_client_tls(SSL* ssl) {
    int client_socket = SSL_get_fd(ssl);
    std::cout << "TLS handshake successful. Socket: " << client_socket << ". Starting SGX Attestation." << std::endl;
    std::string current_client_id = ""; // Unregistered initially for this session

    // --- SGX Attestation Flow ---
    { // Scoping for attestation data vectors
        std::vector<uint8_t> client_quote_vec;
        if (!receive_client_attestation_data(ssl, client_quote_vec)) {
            std::cerr << "Failed to receive client attestation data. Closing connection. Socket: " << client_socket << std::endl;
            goto cleanup_ssl_and_active_id;
        }

        if (!verify_client_quote(client_quote_vec.data(), client_quote_vec.size())) {
            std::cerr << "Client quote verification failed. Closing connection. Socket: " << client_socket << std::endl;
            goto cleanup_ssl_and_active_id;
        }
        std::cout << "Client attestation successful. Socket: " << client_socket << std::endl;

        std::vector<uint8_t> server_quote_vec;
        if (!generate_server_quote(ssl, server_quote_vec)) {
            std::cerr << "Failed to generate server quote. Closing connection. Socket: " << client_socket << std::endl;
            goto cleanup_ssl_and_active_id;
        }
        if (!send_server_attestation_data(ssl, server_quote_vec)) {
            std::cerr << "Failed to send server attestation data. Closing connection. Socket: " << client_socket << std::endl;
            goto cleanup_ssl_and_active_id;
        }
        std::cout << "Server attestation successful. Socket: " << client_socket << std::endl;
    } // End of scoping for attestation data
    std::cout << "Mutual SGX attestation completed. Proceeding to application message loop. Socket: " << client_socket << std::endl;

    // --- Application Message Loop ---
    while (true) {
        uint8_t msg_type;
        std::vector<uint8_t> payload;

        if (!read_message(ssl, msg_type, payload)) {
            std::cout << "Client disconnected or read_message failed. Socket: " << client_socket << std::endl;
            break; // Exit loop to cleanup
        }

        std::cout << "Received message type: 0x" << std::hex << (int)msg_type << std::dec << " from socket " << client_socket << std::endl;

        switch (msg_type) {
            case MSG_TYPE_REGISTER_CLIENT: {
                if (payload.empty()) {
                    std::cerr << "MSG_TYPE_REGISTER_CLIENT: Payload is empty. Socket: " << client_socket << std::endl;
                    send_response(ssl, MSG_TYPE_ERROR, std::vector<uint8_t>(std::string("Empty client ID").begin(), std::string("Empty client ID").end()));
                    continue;
                }
                std::string client_id_str(payload.begin(), payload.end());
                std::cout << "MSG_TYPE_REGISTER_CLIENT: Attempting to register client_id '" << client_id_str << "' for socket " << client_socket << std::endl;
                
                bool success = false;
                {
                    std::lock_guard<std::mutex> lock_active(active_client_ids_mutex_);
                    std::lock_guard<std::mutex> lock_queues(client_message_queues_mutex_);
                    
                    // Optional: Check if client_id_str is already in active_client_ids_ by another SSL session.
                    // This is complex as active_client_ids_ maps SSL* to ID. Would need to iterate.
                    // For simplicity, we register or update. A client can "re-register" on a new connection.
                    // If an old SSL session for this ID exists but is dead, it will be cleaned up eventually.

                    active_client_ids_[ssl] = client_id_str;
                    current_client_id = client_id_str; // Update current session's client_id

                    if (client_message_queues_.find(client_id_str) == client_message_queues_.end()) {
                        client_message_queues_[client_id_str] = std::queue<std::vector<uint8_t>>();
                        std::cout << "Created new message queue for client_id: " << client_id_str << std::endl;
                    }
                    success = true;
                } // Mutexes released

                if (success) {
                    send_simple_response(ssl, MSG_TYPE_REGISTER_ACK);
                    std::cout << "Client_id '" << client_id_str << "' registered successfully for socket " << client_socket << std::endl;
                } else {
                     // This path is less likely with current logic but kept for robustness
                    send_response(ssl, MSG_TYPE_REGISTER_NACK, std::vector<uint8_t>(std::string("Registration failed").begin(), std::string("Registration failed").end()));
                }
                break;
            }

            case MSG_TYPE_SEND_DATA: {
                if (current_client_id.empty()) {
                    std::cerr << "MSG_TYPE_SEND_DATA: Sender not registered. Socket: " << client_socket << std::endl;
                    send_response(ssl, MSG_TYPE_ERROR, std::vector<uint8_t>(std::string("Not registered").begin(), std::string("Not registered").end()));
                    continue;
                }

                // Payload: recipient_id_string_length (uint16_t), recipient_id_string (variable), data_payload (variable)
                if (payload.size() < sizeof(uint16_t)) {
                    std::cerr << "MSG_TYPE_SEND_DATA: Payload too short for recipient_id length. Socket: " << client_socket << std::endl;
                    send_response(ssl, MSG_TYPE_ERROR, std::vector<uint8_t>(std::string("Malformed SEND_DATA").begin(), std::string("Malformed SEND_DATA").end()));
                    continue;
                }

                uint16_t recipient_id_len_net;
                memcpy(&recipient_id_len_net, payload.data(), sizeof(uint16_t));
                uint16_t recipient_id_len = ntohs(recipient_id_len_net);

                if (payload.size() < sizeof(uint16_t) + recipient_id_len) {
                    std::cerr << "MSG_TYPE_SEND_DATA: Payload too short for recipient_id. Socket: " << client_socket << std::endl;
                     send_response(ssl, MSG_TYPE_ERROR, std::vector<uint8_t>(std::string("Malformed SEND_DATA").begin(), std::string("Malformed SEND_DATA").end()));
                    continue;
                }

                std::string recipient_id_str(payload.begin() + sizeof(uint16_t), payload.begin() + sizeof(uint16_t) + recipient_id_len);
                std::vector<uint8_t> data_payload(payload.begin() + sizeof(uint16_t) + recipient_id_len, payload.end());
                
                std::cout << "MSG_TYPE_SEND_DATA: From '" << current_client_id << "' to '" << recipient_id_str << "', data size: " << data_payload.size() << ". Socket: " << client_socket << std::endl;

                bool recipient_found = false;
                {
                    std::lock_guard<std::mutex> lock(client_message_queues_mutex_);
                    auto it = client_message_queues_.find(recipient_id_str);
                    if (it != client_message_queues_.end()) {
                        // Construct message for recipient: sender_id_length, sender_id, data_payload
                        std::vector<uint8_t> message_for_recipient;
                        uint16_t sender_id_len = static_cast<uint16_t>(current_client_id.length());
                        uint16_t sender_id_len_net = htons(sender_id_len);

                        message_for_recipient.insert(message_for_recipient.end(), reinterpret_cast<uint8_t*>(&sender_id_len_net), reinterpret_cast<uint8_t*>(&sender_id_len_net) + sizeof(uint16_t));
                        message_for_recipient.insert(message_for_recipient.end(), current_client_id.begin(), current_client_id.end());
                        message_for_recipient.insert(message_for_recipient.end(), data_payload.begin(), data_payload.end());
                        
                        it->second.push(message_for_recipient);
                        recipient_found = true;
                        std::cout << "Data queued for recipient: " << recipient_id_str << std::endl;
                    } else {
                        std::cout << "Recipient '" << recipient_id_str << "' not found or no queue." << std::endl;
                    }
                } // Mutex released

                if (recipient_found) {
                    send_simple_response(ssl, MSG_TYPE_SEND_ACK);
                } else {
                    send_response(ssl, MSG_TYPE_SEND_NACK, std::vector<uint8_t>(std::string("Recipient not found").begin(), std::string("Recipient not found").end()));
                }
                break;
            }

            case MSG_TYPE_POLL_DATA: {
                 if (current_client_id.empty()) {
                    std::cerr << "MSG_TYPE_POLL_DATA: Client not registered. Socket: " << client_socket << std::endl;
                    send_response(ssl, MSG_TYPE_POLL_NACK, std::vector<uint8_t>(std::string("Not registered").begin(), std::string("Not registered").end()));
                    continue;
                }
                std::cout << "MSG_TYPE_POLL_DATA: Client '" << current_client_id << "' polling for data. Socket: " << client_socket << std::endl;

                std::vector<uint8_t> data_to_send;
                bool data_was_available = false;
                {
                    std::lock_guard<std::mutex> lock(client_message_queues_mutex_);
                    auto it = client_message_queues_.find(current_client_id);
                    if (it != client_message_queues_.end() && !it->second.empty()) {
                        data_to_send = it->second.front();
                        it->second.pop();
                        data_was_available = true;
                        std::cout << "Data found for client '" << current_client_id << "', size: " << data_to_send.size() << std::endl;
                    } else {
                         std::cout << "No data available for client '" << current_client_id << "'." << std::endl;
                    }
                } // Mutex released

                if (data_was_available) {
                    send_response(ssl, MSG_TYPE_DATA_AVAILABLE, data_to_send);
                } else {
                    send_simple_response(ssl, MSG_TYPE_NO_DATA_AVAILABLE);
                }
                break;
            }

            default: {
                std::cerr << "Unknown message type: 0x" << std::hex << (int)msg_type << std::dec << ". Socket: " << client_socket << std::endl;
                std::string err_msg = "Unknown message type: " + std::to_string(msg_type);
                send_response(ssl, MSG_TYPE_ERROR, std::vector<uint8_t>(err_msg.begin(), err_msg.end()));
                break;
            }
        }
    } // End of while(true) message loop

cleanup_ssl_and_active_id:
    // Client disconnected or error occurred, clean up active client ID mapping
    if (!current_client_id.empty()) { // Or check if (ssl was in active_client_ids_)
        std::cout << "Cleaning up active_client_id mapping for client '" << current_client_id << "' (socket " << client_socket << ")" << std::endl;
        std::lock_guard<std::mutex> lock(active_client_ids_mutex_);
        // Only remove if the current SSL session is the one associated with current_client_id
        // This check is implicitly handled if we just remove by SSL* key
        auto it = active_client_ids_.find(ssl);
        if (it != active_client_ids_.end()) {
            if (it->second == current_client_id) { // Additional check, though removing by SSL* is key
                 active_client_ids_.erase(it);
                 std::cout << "Removed SSL session from active_client_ids_ for client_id: " << current_client_id << std::endl;
            } else {
                // This might happen if client re-registers with a new ID on the same SSL session (not typical with current logic)
                // or if current_client_id was not properly updated.
                 std::cout << "Warning: current_client_id '" << current_client_id << "' did not match active_client_ids_ entry '" << it->second << "' for this SSL session. Erasing by SSL* anyway." << std::endl;
                 active_client_ids_.erase(it);
            }
        } else {
            std::cout << "SSL session not found in active_client_ids_ during cleanup for client_id: " << current_client_id << " (socket " << client_socket << "). Might have been cleared already or never fully registered." << std::endl;
        }
    } else {
         std::cout << "No client_id registered for this session (socket " << client_socket << ") or already cleaned up." << std::endl;
         // Still attempt to remove by SSL* just in case it was added without current_client_id being set (should not happen)
         std::lock_guard<std::mutex> lock(active_client_ids_mutex_);
         active_client_ids_.erase(ssl);
    }


    // Graceful SSL shutdown
    int shutdown_ret = SSL_shutdown(ssl);
    if (shutdown_ret == 0) {
        shutdown_ret = SSL_shutdown(ssl); 
    }
     if (shutdown_ret < 0 && SSL_get_error(ssl, shutdown_ret) != SSL_ERROR_ZERO_RETURN && SSL_get_error(ssl, shutdown_ret) != SSL_ERROR_SYSCALL) {
         std::cerr << "SSL_shutdown failed for client " << client_socket << ". Error: " << SSL_get_error(ssl, shutdown_ret) << std::endl;
         ERR_print_errors_fp(stderr);
    }

    SSL_free(ssl);
    close(client_socket);
    std::cout << "Client connection closed, resources freed. Socket: " << client_socket << std::endl;
}
