#pragma once
#include "sdse_protocol.h" // For ParsedMessage and constants
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <thread>
#include <atomic>

class SdseServer {
public:
    SdseServer(const std::string& uds_socket_path);
    ~SdseServer();

    bool start();
    void stop(); // For graceful shutdown

private:
    void accept_loop();
    void handle_client_connection(int client_fd);

    // Message handling helpers
    bool read_sdse_message(int fd, SdseParsedMessage& msg);
    bool send_sdse_response(int fd, uint8_t msg_type, uint8_t flags, 
                            const std::vector<uint8_t>& payload, 
                            uint32_t request_id = 0); // request_id not used yet but in signature
    bool send_simple_status_response(int fd, uint8_t msg_type, uint8_t status, uint8_t original_flags);


    // Command handlers
    void handle_register_client(int client_fd, const SdseParsedMessage& msg);
    void handle_store_data(int client_fd, const SdseParsedMessage& msg);
    void handle_retrieve_data(int client_fd, const SdseParsedMessage& msg);
    void handle_delete_data(int client_fd, const SdseParsedMessage& msg);

    // Utility
    void log_audit(const std::string& message);
    std::string bytes_to_hex_string(const std::vector<uint8_t>& bytes);
    bool is_client_registered(int client_fd, std::vector<uint8_t>& client_id_out); // Helper to check and get client_id

    // Member variables
    std::string uds_socket_path_;
    int listener_fd_;
    std::atomic<bool> running_;
    std::thread listener_thread_;
    std::vector<std::thread> client_handler_threads_; // To manage client threads for joining

    // Data storage
    std::map<std::string, std::vector<uint8_t>> data_store_; // object_id (string) -> data
    std::mutex data_store_mutex_;

    // Client registration & session management
    std::map<int, std::vector<uint8_t>> registered_clients_; // client_fd -> client_id_hash (MRENCLAVE)
    std::mutex clients_mutex_;

    // Audit log
    std::vector<std::string> audit_log_;
    std::mutex audit_log_mutex_;
};
