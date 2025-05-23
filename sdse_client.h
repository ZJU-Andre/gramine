#pragma once
#include "sdse_protocol.h" // For SdseParsedMessage and constants
#include <string>
#include <vector>
#include <cstdint> // For uint8_t, uint32_t

class SdseClient {
public:
    SdseClient(const std::string& uds_socket_path);
    ~SdseClient();

    bool connect_to_sdse();
    void disconnect_from_sdse();

    // API Functions
    bool register_client(const std::vector<uint8_t>& client_id_hash);
    bool store_data(const std::string& object_id, const std::vector<uint8_t>& data, bool request_ack = true);
    bool retrieve_data(const std::string& object_id, std::vector<uint8_t>& out_data);
    bool delete_data(const std::string& object_id, bool request_ack = true);

private:
    // Helper methods for sending requests and receiving responses
    bool send_sdse_request(uint8_t msg_type, uint8_t flags, const std::vector<uint8_t>& payload, uint32_t request_id = 0);
    bool receive_sdse_response(SdseParsedMessage& parsed_resp);

    // Private members
    std::string uds_socket_path_;
    int client_fd_ = -1;
};
