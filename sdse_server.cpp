#include "sdse_server.h"
#include <iostream>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>
#include <arpa/inet.h> // For ntohl, htonl
#include <algorithm>   // For std::remove
#include <iomanip>     // For std::hex, std::setw, std::setfill
#include <sstream>     // For stringstream in hex conversion

// Helper to convert byte vector to hex string for logging/client ID
std::string SdseServer::bytes_to_hex_string(const std::vector<uint8_t>& bytes) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (uint8_t byte : bytes) {
        ss << std::setw(2) << static_cast<int>(byte);
    }
    return ss.str();
}

SdseServer::SdseServer(const std::string& uds_socket_path)
    : uds_socket_path_(uds_socket_path), listener_fd_(-1), running_(false) {}

SdseServer::~SdseServer() {
    stop(); // Ensure server is stopped and resources are cleaned up
}

void SdseServer::log_audit(const std::string& message) {
    std::lock_guard<std::mutex> lock(audit_log_mutex_);
    // In a real system, this would write to a secure, persistent log.
    // For now, just print and store in memory (if needed for inspection).
    std::cout << "[AUDIT] " << message << std::endl;
    audit_log_.push_back(message); 
}

bool SdseServer::start() {
    if (running_) {
        std::cerr << "SDSE Server already running." << std::endl;
        return true;
    }

    listener_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listener_fd_ < 0) {
        perror("socket error");
        return false;
    }

    // Remove existing socket file if it exists
    unlink(uds_socket_path_.c_str());

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, uds_socket_path_.c_str(), sizeof(addr.sun_path) - 1);

    if (bind(listener_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("bind error");
        close(listener_fd_);
        listener_fd_ = -1;
        return false;
    }

    if (listen(listener_fd_, 5) < 0) { // Backlog of 5
        perror("listen error");
        close(listener_fd_);
        listener_fd_ = -1;
        return false;
    }

    running_ = true;
    listener_thread_ = std::thread(&SdseServer::accept_loop, this);
    std::cout << "SDSE Server started on UDS: " << uds_socket_path_ << std::endl;
    log_audit("Server started on UDS: " + uds_socket_path_);
    return true;
}

void SdseServer::stop() {
    if (!running_) {
        return;
    }
    running_ = false;

    // Close listener socket to unblock accept()
    if (listener_fd_ != -1) {
        //shutdown(listener_fd_, SHUT_RDWR); // Cause accept to return
        close(listener_fd_); // This should also cause accept to return with an error
        listener_fd_ = -1;
    }

    if (listener_thread_.joinable()) {
        listener_thread_.join();
    }
    
    // TODO: Gracefully close active client connections and join their threads
    // For now, client threads are detached and will exit when connection closes or error.
    // To join them, would need to store thread objects and signal them to stop.
    // For simplicity in this phase, not implemented.

    std::cout << "SDSE Server stopped." << std::endl;
    log_audit("Server stopped.");
}

void SdseServer::accept_loop() {
    while (running_) {
        int client_fd = accept(listener_fd_, nullptr, nullptr);
        if (client_fd < 0) {
            if (running_) { // Only log error if we weren't intentionally stopping
                perror("accept error");
            }
            // If accept failed (e.g. listener_fd_ closed by stop()), break loop
            if (!running_) break; 
            continue; 
        }
        std::cout << "Accepted new client connection. FD: " << client_fd << std::endl;
        log_audit("Accepted new client connection. FD: " + std::to_string(client_fd));
        
        // Spawn a new thread to handle this client
        // TODO: Manage these threads for graceful shutdown (e.g., store in a list and join in stop())
        std::thread client_thread(&SdseServer::handle_client_connection, this, client_fd);
        client_thread.detach(); // For now, detach.
    }
    std::cout << "Accept loop finished." << std::endl;
}


bool SdseServer::read_sdse_message(int fd, SdseParsedMessage& msg) {
    msg = {}; // Clear previous message state

    std::vector<uint8_t> header_buf(SDSE_HEADER_SIZE_BASE);
    ssize_t bytes_read = read(fd, header_buf.data(), SDSE_HEADER_SIZE_BASE);

    if (bytes_read == 0) { // Connection closed by client
        std::cout << "Client FD " << fd << " disconnected (EOF)." << std::endl;
        return false;
    }
    if (bytes_read < 0) {
        perror("read error on header");
        return false;
    }
    if (bytes_read != SDSE_HEADER_SIZE_BASE) {
        std::cerr << "Failed to read full header. Expected " << SDSE_HEADER_SIZE_BASE << " got " << bytes_read << std::endl;
        return false;
    }

    msg.msg_type = header_buf[0];
    msg.flags = header_buf[1];
    memcpy(&msg.payload_len, &header_buf[2], sizeof(uint32_t));
    msg.payload_len = ntohl(msg.payload_len);

    // Simple logic: REGISTER_CLIENT_REQ has client ID in payload for this version.
    // Other messages might imply client ID from session (fd) after registration.
    // For this subtask, we assume the client_id is part of the payload for REGISTER_CLIENT_REQ
    // and not explicitly sent in the header for other messages.
    // RequestID also not used in this simple version.

    if (msg.payload_len > 0) {
        // Max payload size check
        const uint32_t MAX_PAYLOAD_LIMIT = 1024 * 1024 * 5; // 5MB limit for example
        if (msg.payload_len > MAX_PAYLOAD_LIMIT) {
            std::cerr << "Payload length " << msg.payload_len << " exceeds limit " << MAX_PAYLOAD_LIMIT << std::endl;
            return false;
        }
        msg.payload.resize(msg.payload_len);
        bytes_read = read(fd, msg.payload.data(), msg.payload_len);
        if (bytes_read < 0) {
            perror("read error on payload");
            return false;
        }
        if (static_cast<uint32_t>(bytes_read) != msg.payload_len) {
            std::cerr << "Failed to read full payload. Expected " << msg.payload_len << " got " << bytes_read << std::endl;
            return false;
        }
    }
    return true;
}

bool SdseServer::send_sdse_response(int fd, uint8_t msg_type, uint8_t flags, 
                                   const std::vector<uint8_t>& payload, 
                                   uint32_t request_id) {
    std::vector<uint8_t> response_buf;
    response_buf.push_back(msg_type);
    response_buf.push_back(flags);

    uint32_t payload_len_nbo = htonl(static_cast<uint32_t>(payload.size()));
    response_buf.insert(response_buf.end(), 
                        reinterpret_cast<uint8_t*>(&payload_len_nbo), 
                        reinterpret_cast<uint8_t*>(&payload_len_nbo) + sizeof(uint32_t));
    
    // RequestID not used in this simple version's framing for responses.

    response_buf.insert(response_buf.end(), payload.begin(), payload.end());

    ssize_t bytes_sent = write(fd, response_buf.data(), response_buf.size());
    if (bytes_sent < 0) {
        perror("write error on response");
        return false;
    }
    if (static_cast<size_t>(bytes_sent) != response_buf.size()) {
        std::cerr << "Failed to send full response. Expected " << response_buf.size() << " sent " << bytes_sent << std::endl;
        return false;
    }
    return true;
}

bool SdseServer::send_simple_status_response(int fd, uint8_t msg_type, uint8_t status, uint8_t original_flags) {
    // Check if original request wanted an ACK. If not, maybe don't send one for some operations.
    // For now, we send if the response type implies it (e.g. _RESP types).
    // The `original_flags & SDSE_FLAG_REQUEST_ACK` check can be done by the caller.
    std::vector<uint8_t> status_payload = {status};
    return send_sdse_response(fd, msg_type, SDSE_FLAG_NONE, status_payload);
}

bool SdseServer::is_client_registered(int client_fd, std::vector<uint8_t>& client_id_out) {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    auto it = registered_clients_.find(client_fd);
    if (it != registered_clients_.end()) {
        client_id_out = it->second;
        return true;
    }
    return false;
}


void SdseServer::handle_client_connection(int client_fd) {
    SdseParsedMessage msg;
    std::vector<uint8_t> current_client_id_hash; // Store MRENCLAVE hash after registration

    log_audit("Handling new client connection. FD: " + std::to_string(client_fd));

    while (running_ && read_sdse_message(client_fd, msg)) {
        std::cout << "Received message from FD " << client_fd << ": Type 0x" << std::hex << (int)msg.msg_type 
                  << ", Flags 0x" << (int)msg.flags << ", PayloadLen " << std::dec << msg.payload_len << std::endl;

        switch (msg.msg_type) {
            case SDSE_MSG_TYPE_REGISTER_CLIENT_REQ:
                handle_register_client(client_fd, msg);
                break;
            case SDSE_MSG_TYPE_STORE_DATA_REQ:
                handle_store_data(client_fd, msg);
                break;
            case SDSE_MSG_TYPE_RETRIEVE_DATA_REQ:
                handle_retrieve_data(client_fd, msg);
                break;
            case SDSE_MSG_TYPE_DELETE_DATA_REQ:
                handle_delete_data(client_fd, msg);
                break;
            default:
                std::cerr << "Unknown message type: 0x" << std::hex << (int)msg.msg_type << std::dec << std::endl;
                log_audit("Unknown message type from FD " + std::to_string(client_fd) + ": 0x" + bytes_to_hex_string({msg.msg_type}));
                send_simple_status_response(client_fd, SDSE_MSG_TYPE_ERROR_RESP, SDSE_STATUS_ERROR_INVALID_REQUEST, msg.flags);
                break;
        }
    }

    // Client disconnected or error
    std::cout << "Client FD " << client_fd << " connection ended." << std::endl;
    log_audit("Client FD " + std::to_string(client_fd) + " disconnected.");
    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        registered_clients_.erase(client_fd);
    }
    close(client_fd);
}

void SdseServer::handle_register_client(int client_fd, const SdseParsedMessage& msg) {
    if (msg.payload.size() != SDSE_CLIENT_ID_SIZE) {
        std::cerr << "REGISTER_CLIENT_REQ: Invalid payload size for client ID. Expected " 
                  << SDSE_CLIENT_ID_SIZE << " got " << msg.payload.size() << std::endl;
        log_audit("REGISTER_CLIENT_REQ failed from FD " + std::to_string(client_fd) + ": Invalid ClientID payload size.");
        send_simple_status_response(client_fd, SDSE_MSG_TYPE_REGISTER_CLIENT_RESP, SDSE_STATUS_ERROR_INVALID_REQUEST, msg.flags);
        return;
    }
    
    std::string client_id_hex = bytes_to_hex_string(msg.payload);
    log_audit("REGISTER_CLIENT_REQ from FD " + std::to_string(client_fd) + " with ClientID (MRENCLAVE_HASH): " + client_id_hex);

    {
        std::lock_guard<std::mutex> lock(clients_mutex_);
        // Optional: Check if this MRENCLAVE is already registered by another FD (more complex logic)
        // For now, simple registration by FD. If FD reconnects, it needs to re-register.
        registered_clients_[client_fd] = msg.payload;
    }
    
    std::cout << "Client registered. FD: " << client_fd << ", ClientID (Hash): " << client_id_hex << std::endl;
    send_simple_status_response(client_fd, SDSE_MSG_TYPE_REGISTER_CLIENT_RESP, SDSE_STATUS_OK, msg.flags);
}

void SdseServer::handle_store_data(int client_fd, const SdseParsedMessage& msg) {
    std::vector<uint8_t> client_id_hash;
    if (!is_client_registered(client_fd, client_id_hash)) {
        log_audit("STORE_DATA_REQ failed from FD " + std::to_string(client_fd) + ": Client not registered.");
        send_simple_status_response(client_fd, SDSE_MSG_TYPE_ERROR_RESP, SDSE_STATUS_ERROR_NOT_REGISTERED, msg.flags);
        return;
    }

    // Payload: object_id_len (uint16_t, NBO) + object_id (string) + data_chunk
    if (msg.payload.size() < sizeof(uint16_t)) {
        log_audit("STORE_DATA_REQ failed from FD " + std::to_string(client_fd) + ": Payload too short for object_id_len.");
        send_simple_status_response(client_fd, SDSE_MSG_TYPE_STORE_DATA_RESP, SDSE_STATUS_ERROR_INVALID_REQUEST, msg.flags);
        return;
    }
    uint16_t object_id_len_nbo;
    memcpy(&object_id_len_nbo, msg.payload.data(), sizeof(uint16_t));
    uint16_t object_id_len = ntohs(object_id_len_nbo);

    if (msg.payload.size() < sizeof(uint16_t) + object_id_len) {
        log_audit("STORE_DATA_REQ failed from FD " + std::to_string(client_fd) + ": Payload too short for object_id.");
        send_simple_status_response(client_fd, SDSE_MSG_TYPE_STORE_DATA_RESP, SDSE_STATUS_ERROR_INVALID_REQUEST, msg.flags);
        return;
    }
    std::string object_id(msg.payload.begin() + sizeof(uint16_t), msg.payload.begin() + sizeof(uint16_t) + object_id_len);
    std::vector<uint8_t> data_chunk(msg.payload.begin() + sizeof(uint16_t) + object_id_len, msg.payload.end());

    // Placeholder ACL: Any registered client can store. A real system would check client_id_hash against object_id ownership/permissions.
    log_audit("STORE_DATA_REQ from FD " + std::to_string(client_fd) + " (ClientID: " + bytes_to_hex_string(client_id_hash) + 
              ") for Object ID: " + object_id + ", Data Size: " + std::to_string(data_chunk.size()));
    
    {
        std::lock_guard<std::mutex> lock(data_store_mutex_);
        data_store_[object_id] = data_chunk;
    }
    std::cout << "Stored data for object: " << object_id << std::endl;

    if (msg.flags & SDSE_FLAG_REQUEST_ACK) {
        send_simple_status_response(client_fd, SDSE_MSG_TYPE_STORE_DATA_RESP, SDSE_STATUS_OK, msg.flags);
    }
}

void SdseServer::handle_retrieve_data(int client_fd, const SdseParsedMessage& msg) {
    std::vector<uint8_t> client_id_hash;
    if (!is_client_registered(client_fd, client_id_hash)) {
        log_audit("RETRIEVE_DATA_REQ failed from FD " + std::to_string(client_fd) + ": Client not registered.");
        send_simple_status_response(client_fd, SDSE_MSG_TYPE_RETRIEVE_DATA_NACK, SDSE_STATUS_ERROR_NOT_REGISTERED, msg.flags);
        return;
    }
    
    // Payload: object_id_len (uint16_t, NBO) + object_id (string)
     if (msg.payload.size() < sizeof(uint16_t)) {
        log_audit("RETRIEVE_DATA_REQ failed from FD " + std::to_string(client_fd) + ": Payload too short for object_id_len.");
        send_simple_status_response(client_fd, SDSE_MSG_TYPE_RETRIEVE_DATA_NACK, SDSE_STATUS_ERROR_INVALID_REQUEST, msg.flags);
        return;
    }
    uint16_t object_id_len_nbo;
    memcpy(&object_id_len_nbo, msg.payload.data(), sizeof(uint16_t));
    uint16_t object_id_len = ntohs(object_id_len_nbo);

    if (msg.payload.size() < sizeof(uint16_t) + object_id_len) {
        log_audit("RETRIEVE_DATA_REQ failed from FD " + std::to_string(client_fd) + ": Payload too short for object_id.");
        send_simple_status_response(client_fd, SDSE_MSG_TYPE_RETRIEVE_DATA_NACK, SDSE_STATUS_ERROR_INVALID_REQUEST, msg.flags);
        return;
    }
    std::string object_id(msg.payload.begin() + sizeof(uint16_t), msg.payload.begin() + sizeof(uint16_t) + object_id_len);

    log_audit("RETRIEVE_DATA_REQ from FD " + std::to_string(client_fd) + " (ClientID: " + bytes_to_hex_string(client_id_hash) + 
              ") for Object ID: " + object_id);

    std::vector<uint8_t> data_chunk;
    bool found = false;
    {
        std::lock_guard<std::mutex> lock(data_store_mutex_);
        auto it = data_store_.find(object_id);
        if (it != data_store_.end()) {
            data_chunk = it->second;
            found = true;
        }
    }

    if (found) {
        // Placeholder ACL: Any registered client can retrieve.
        std::cout << "Retrieved data for object: " << object_id << ", Size: " << data_chunk.size() << std::endl;
        // For RETRIEVE_DATA_RESP, payload is: status (1B) + actual_data_payload
        std::vector<uint8_t> response_payload;
        response_payload.push_back(SDSE_STATUS_OK);
        response_payload.insert(response_payload.end(), data_chunk.begin(), data_chunk.end());
        send_sdse_response(client_fd, SDSE_MSG_TYPE_RETRIEVE_DATA_RESP, SDSE_FLAG_NONE, response_payload);
    } else {
        std::cout << "Object not found: " << object_id << std::endl;
        send_simple_status_response(client_fd, SDSE_MSG_TYPE_RETRIEVE_DATA_NACK, SDSE_STATUS_ERROR_NOT_FOUND, msg.flags);
    }
}

void SdseServer::handle_delete_data(int client_fd, const SdseParsedMessage& msg) {
    std::vector<uint8_t> client_id_hash;
    if (!is_client_registered(client_fd, client_id_hash)) {
        log_audit("DELETE_DATA_REQ failed from FD " + std::to_string(client_fd) + ": Client not registered.");
        send_simple_status_response(client_fd, SDSE_MSG_TYPE_ERROR_RESP, SDSE_STATUS_ERROR_NOT_REGISTERED, msg.flags);
        return;
    }

    // Payload: object_id_len (uint16_t, NBO) + object_id (string)
    if (msg.payload.size() < sizeof(uint16_t)) {
        log_audit("DELETE_DATA_REQ failed from FD " + std::to_string(client_fd) + ": Payload too short for object_id_len.");
        send_simple_status_response(client_fd, SDSE_MSG_TYPE_DELETE_DATA_RESP, SDSE_STATUS_ERROR_INVALID_REQUEST, msg.flags);
        return;
    }
    uint16_t object_id_len_nbo;
    memcpy(&object_id_len_nbo, msg.payload.data(), sizeof(uint16_t));
    uint16_t object_id_len = ntohs(object_id_len_nbo);

    if (msg.payload.size() < sizeof(uint16_t) + object_id_len) {
        log_audit("DELETE_DATA_REQ failed from FD " + std::to_string(client_fd) + ": Payload too short for object_id.");
        send_simple_status_response(client_fd, SDSE_MSG_TYPE_DELETE_DATA_RESP, SDSE_STATUS_ERROR_INVALID_REQUEST, msg.flags);
        return;
    }
    std::string object_id(msg.payload.begin() + sizeof(uint16_t), msg.payload.begin() + sizeof(uint16_t) + object_id_len);
    
    // Placeholder ACL: Any registered client can delete.
    log_audit("DELETE_DATA_REQ from FD " + std::to_string(client_fd) + " (ClientID: " + bytes_to_hex_string(client_id_hash) + 
              ") for Object ID: " + object_id);

    size_t erased_count = 0;
    {
        std::lock_guard<std::mutex> lock(data_store_mutex_);
        erased_count = data_store_.erase(object_id);
    }

    if (erased_count > 0) {
        std::cout << "Deleted object: " << object_id << std::endl;
        if (msg.flags & SDSE_FLAG_REQUEST_ACK) {
            send_simple_status_response(client_fd, SDSE_MSG_TYPE_DELETE_DATA_RESP, SDSE_STATUS_OK, msg.flags);
        }
    } else {
        std::cout << "Attempted to delete non-existent object: " << object_id << std::endl;
        if (msg.flags & SDSE_FLAG_REQUEST_ACK) {
            send_simple_status_response(client_fd, SDSE_MSG_TYPE_DELETE_DATA_RESP, SDSE_STATUS_ERROR_NOT_FOUND, msg.flags);
        }
    }
}
