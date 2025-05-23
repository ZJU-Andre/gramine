#include "sdse_client.h"
#include "sdse_protocol.h" // For message types and constants
#include <iostream>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <vector>
#include <arpa/inet.h> // For ntohl, htonl
#include <cstring>     // For memcpy, strncpy, strerror
#include <algorithm>   // For std::min

SdseClient::SdseClient(const std::string& uds_socket_path)
    : uds_socket_path_(uds_socket_path), client_fd_(-1) {}

SdseClient::~SdseClient() {
    disconnect_from_sdse();
}

bool SdseClient::connect_to_sdse() {
    if (client_fd_ != -1) {
        std::cerr << "Already connected." << std::endl;
        return true;
    }

    client_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (client_fd_ < 0) {
        perror("socket error (client)");
        return false;
    }

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, uds_socket_path_.c_str(), sizeof(addr.sun_path) - 1);

    if (connect(client_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("connect error (client)");
        close(client_fd_);
        client_fd_ = -1;
        return false;
    }

    std::cout << "Successfully connected to SDSE server at " << uds_socket_path_ << std::endl;
    return true;
}

void SdseClient::disconnect_from_sdse() {
    if (client_fd_ != -1) {
        std::cout << "Disconnecting from SDSE server." << std::endl;
        close(client_fd_);
        client_fd_ = -1;
    }
}

bool SdseClient::send_sdse_request(uint8_t msg_type, uint8_t flags, const std::vector<uint8_t>& payload, uint32_t request_id) {
    if (client_fd_ == -1) {
        std::cerr << "Not connected to SDSE server." << std::endl;
        return false;
    }

    std::vector<uint8_t> request_buf;
    request_buf.push_back(msg_type);
    request_buf.push_back(flags);

    uint32_t payload_len_nbo = htonl(static_cast<uint32_t>(payload.size()));
    request_buf.insert(request_buf.end(),
                       reinterpret_cast<uint8_t*>(&payload_len_nbo),
                       reinterpret_cast<uint8_t*>(&payload_len_nbo) + sizeof(uint32_t));

    // Note: ClientID and RequestID are not part of the base header in this simplified protocol version.
    // ClientID is in payload for REGISTER_CLIENT_REQ. RequestID is not used.

    request_buf.insert(request_buf.end(), payload.begin(), payload.end());

    ssize_t bytes_sent = write(client_fd_, request_buf.data(), request_buf.size());
    if (bytes_sent < 0) {
        perror("write error on request (client)");
        return false;
    }
    if (static_cast<size_t>(bytes_sent) != request_buf.size()) {
        std::cerr << "Failed to send full request. Expected " << request_buf.size() << " sent " << bytes_sent << std::endl;
        return false;
    }
    return true;
}

bool SdseClient::receive_sdse_response(SdseParsedMessage& parsed_resp) {
    if (client_fd_ == -1) {
        std::cerr << "Not connected to SDSE server (cannot receive)." << std::endl;
        return false;
    }
    parsed_resp = {}; // Clear previous state

    std::vector<uint8_t> header_buf(SDSE_HEADER_SIZE_BASE);
    ssize_t bytes_read = read(client_fd_, header_buf.data(), SDSE_HEADER_SIZE_BASE);

    if (bytes_read == 0) {
        std::cout << "Server disconnected (EOF)." << std::endl;
        return false;
    }
    if (bytes_read < 0) {
        perror("read error on response header (client)");
        return false;
    }
    if (bytes_read != SDSE_HEADER_SIZE_BASE) {
        std::cerr << "Failed to read full response header. Expected " << SDSE_HEADER_SIZE_BASE << " got " << bytes_read << std::endl;
        return false;
    }

    parsed_resp.msg_type = header_buf[0];
    parsed_resp.flags = header_buf[1]; // Flags from server might be useful (e.g. if server can batch/chunk)
    memcpy(&parsed_resp.payload_len, &header_buf[2], sizeof(uint32_t));
    parsed_resp.payload_len = ntohl(parsed_resp.payload_len);

    if (parsed_resp.payload_len > 0) {
        const uint32_t MAX_PAYLOAD_LIMIT = 1024 * 1024 * 5; // 5MB limit
        if (parsed_resp.payload_len > MAX_PAYLOAD_LIMIT) {
            std::cerr << "Response payload length " << parsed_resp.payload_len << " exceeds limit " << MAX_PAYLOAD_LIMIT << std::endl;
            return false;
        }
        parsed_resp.payload.resize(parsed_resp.payload_len);
        bytes_read = read(client_fd_, parsed_resp.payload.data(), parsed_resp.payload_len);
        if (bytes_read < 0) {
            perror("read error on response payload (client)");
            return false;
        }
        if (static_cast<uint32_t>(bytes_read) != parsed_resp.payload_len) {
            std::cerr << "Failed to read full response payload. Expected " << parsed_resp.payload_len << " got " << bytes_read << std::endl;
            return false;
        }
    }
    return true;
}


bool SdseClient::register_client(const std::vector<uint8_t>& client_id_hash) {
    if (client_id_hash.size() != SDSE_CLIENT_ID_SIZE) {
        std::cerr << "Client ID hash must be " << SDSE_CLIENT_ID_SIZE << " bytes." << std::endl;
        return false;
    }
    if (!send_sdse_request(SDSE_MSG_TYPE_REGISTER_CLIENT_REQ, SDSE_FLAG_NONE, client_id_hash)) {
        return false;
    }

    SdseParsedMessage response;
    if (!receive_sdse_response(response)) {
        return false;
    }

    if (response.msg_type == SDSE_MSG_TYPE_REGISTER_CLIENT_RESP && 
        !response.payload.empty() && response.payload[0] == SDSE_STATUS_OK) {
        std::cout << "Client registration successful." << std::endl;
        return true;
    } else {
        std::cerr << "Client registration failed. Server response type: 0x" << std::hex << (int)response.msg_type;
        if (!response.payload.empty()) {
             std::cerr << ", Status: 0x" << (int)response.payload[0];
        }
        std::cerr << std::dec << std::endl;
        return false;
    }
}

bool SdseClient::store_data(const std::string& object_id, const std::vector<uint8_t>& data, bool request_ack) {
    if (object_id.length() > 255) { // Max object_id_len is 1 byte
        std::cerr << "Object ID too long (max 255 bytes)." << std::endl;
        return false;
    }

    std::vector<uint8_t> payload;
    uint8_t object_id_len = static_cast<uint8_t>(object_id.length());
    payload.push_back(object_id_len);
    payload.insert(payload.end(), object_id.begin(), object_id.end());
    
    // Data chunk length (4B NBO) - This was a deviation from server's STORE_DATA_REQ parsing.
    // Server expects object_id_len(uint16_t), object_id, data_chunk.
    // Client was sending object_id_len(1B), object_id, data_chunk_len(4B), data_chunk.
    // Let's align with the server's expectation for STORE_DATA payload:
    // Payload: object_id_len (uint16_t, NBO) + object_id (string) + data_chunk
    // Correcting client's store_data payload construction:
    payload.clear(); // Start over
    uint16_t obj_id_len_nbo = htons(static_cast<uint16_t>(object_id.length()));
    payload.insert(payload.end(), reinterpret_cast<uint8_t*>(&obj_id_len_nbo), reinterpret_cast<uint8_t*>(&obj_id_len_nbo) + sizeof(uint16_t));
    payload.insert(payload.end(), object_id.begin(), object_id.end());
    payload.insert(payload.end(), data.begin(), data.end()); // Data directly follows object_id

    uint8_t flags = SDSE_FLAG_NONE;
    if (request_ack) {
        flags |= SDSE_FLAG_REQUEST_ACK;
    }

    if (!send_sdse_request(SDSE_MSG_TYPE_STORE_DATA_REQ, flags, payload)) {
        return false;
    }

    if (request_ack) {
        SdseParsedMessage response;
        if (!receive_sdse_response(response)) {
            return false;
        }
        if (response.msg_type == SDSE_MSG_TYPE_STORE_DATA_RESP && 
            !response.payload.empty() && response.payload[0] == SDSE_STATUS_OK) {
            std::cout << "Store data for object '" << object_id << "' acknowledged by server." << std::endl;
            return true;
        } else {
            std::cerr << "Store data failed or not acknowledged correctly. Server response type: 0x" << std::hex << (int)response.msg_type;
             if (!response.payload.empty()) {
                 std::cerr << ", Status: 0x" << (int)response.payload[0];
            }
            std::cerr << std::dec << std::endl;
            return false;
        }
    }
    return true; // If no ACK requested, success after send
}

bool SdseClient::retrieve_data(const std::string& object_id, std::vector<uint8_t>& out_data) {
    out_data.clear();
    if (object_id.length() > 255) {
        std::cerr << "Object ID too long." << std::endl;
        return false;
    }
    // Payload: object_id_len (uint16_t, NBO) + object_id (string)
    std::vector<uint8_t> payload;
    uint16_t obj_id_len_nbo = htons(static_cast<uint16_t>(object_id.length()));
    payload.insert(payload.end(), reinterpret_cast<uint8_t*>(&obj_id_len_nbo), reinterpret_cast<uint8_t*>(&obj_id_len_nbo) + sizeof(uint16_t));
    payload.insert(payload.end(), object_id.begin(), object_id.end());
    

    if (!send_sdse_request(SDSE_MSG_TYPE_RETRIEVE_DATA_REQ, SDSE_FLAG_NONE, payload)) {
        return false;
    }

    SdseParsedMessage response;
    if (!receive_sdse_response(response)) {
        return false;
    }

    if (response.msg_type == SDSE_MSG_TYPE_RETRIEVE_DATA_RESP) {
        if (response.payload.empty() || response.payload[0] != SDSE_STATUS_OK) {
            std::cerr << "Retrieve data failed. Server status: 0x" << std::hex 
                      << (response.payload.empty() ? -1 : (int)response.payload[0]) << std::dec << std::endl;
            return false;
        }
        // Data is after the status byte
        out_data.assign(response.payload.begin() + 1, response.payload.end());
        std::cout << "Retrieved data for object '" << object_id << "'. Size: " << out_data.size() << std::endl;
        return true;
    } else if (response.msg_type == SDSE_MSG_TYPE_RETRIEVE_DATA_NACK) {
         std::cerr << "Retrieve data NACK. Server status: 0x" << std::hex 
                   << (response.payload.empty() ? -1 : (int)response.payload[0]) << std::dec << std::endl;
        return false;
    } else {
        std::cerr << "Unexpected response type for retrieve: 0x" << std::hex << (int)response.msg_type << std::dec << std::endl;
        return false;
    }
}

bool SdseClient::delete_data(const std::string& object_id, bool request_ack) {
     if (object_id.length() > 255) {
        std::cerr << "Object ID too long." << std::endl;
        return false;
    }
    // Payload: object_id_len (uint16_t, NBO) + object_id (string)
    std::vector<uint8_t> payload;
    uint16_t obj_id_len_nbo = htons(static_cast<uint16_t>(object_id.length()));
    payload.insert(payload.end(), reinterpret_cast<uint8_t*>(&obj_id_len_nbo), reinterpret_cast<uint8_t*>(&obj_id_len_nbo) + sizeof(uint16_t));
    payload.insert(payload.end(), object_id.begin(), object_id.end());

    uint8_t flags = SDSE_FLAG_NONE;
    if (request_ack) {
        flags |= SDSE_FLAG_REQUEST_ACK;
    }

    if (!send_sdse_request(SDSE_MSG_TYPE_DELETE_DATA_REQ, flags, payload)) {
        return false;
    }

    if (request_ack) {
        SdseParsedMessage response;
        if (!receive_sdse_response(response)) {
            return false;
        }
        if (response.msg_type == SDSE_MSG_TYPE_DELETE_DATA_RESP && 
            !response.payload.empty() && 
            (response.payload[0] == SDSE_STATUS_OK || response.payload[0] == SDSE_STATUS_ERROR_NOT_FOUND) ) {
            std::cout << "Delete data for object '" << object_id << "' acknowledged by server. Status: 0x" 
                      << std::hex << (int)response.payload[0] << std::dec << std::endl;
            return (response.payload[0] == SDSE_STATUS_OK); // True if actually deleted, false if not found but acked
        } else {
            std::cerr << "Delete data failed or not acknowledged correctly. Server response type: 0x" << std::hex << (int)response.msg_type;
            if (!response.payload.empty()) {
                 std::cerr << ", Status: 0x" << (int)response.payload[0];
            }
            std::cerr << std::dec << std::endl;
            return false;
        }
    }
    return true; // If no ACK requested
}
