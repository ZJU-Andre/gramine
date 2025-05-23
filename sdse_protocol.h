#pragma once // Using pragma once for include guard, common in C++
#include <cstdint>
#include <vector>
#include <string>

// Message Framing: | MsgType (1B) | Flags (1B) | PayloadLen (4B, NBO) | [Opt ClientID (32B)] | [Opt ReqID (4B, NBO)] | Payload |
// For simplicity in this initial version, ClientID and ReqID presence will be determined by message type, not flags.
const uint8_t SDSE_HEADER_SIZE_BASE = 1 + 1 + 4; // MsgType + Flags + PayloadLen
const uint8_t SDSE_CLIENT_ID_SIZE = 32;      // Example: SHA256 hash size for MRENCLAVE
const uint8_t SDSE_REQUEST_ID_SIZE = 4;

// Flags (1 Byte) - Bitmask
const uint8_t SDSE_FLAG_NONE = 0x00;
const uint8_t SDSE_FLAG_IS_BATCHED = 0x01;     // Not used in this subtask
const uint8_t SDSE_FLAG_MORE_CHUNKS = 0x02;    // Not used in this subtask
const uint8_t SDSE_FLAG_REQUEST_ACK = 0x04;    // Indicates client wants a response/acknowledgement for operations like STORE/DELETE

// Message Types
// Client Requests
const uint8_t SDSE_MSG_TYPE_REGISTER_CLIENT_REQ = 0x01;
const uint8_t SDSE_MSG_TYPE_STORE_DATA_REQ = 0x02;
const uint8_t SDSE_MSG_TYPE_RETRIEVE_DATA_REQ = 0x03;
const uint8_t SDSE_MSG_TYPE_DELETE_DATA_REQ = 0x04;
// Server Responses
const uint8_t SDSE_MSG_TYPE_REGISTER_CLIENT_RESP = 0x81;
const uint8_t SDSE_MSG_TYPE_STORE_DATA_RESP = 0x82;       // ACK for store if requested
const uint8_t SDSE_MSG_TYPE_RETRIEVE_DATA_RESP = 0x83;    // Contains data
const uint8_t SDSE_MSG_TYPE_RETRIEVE_DATA_NACK = 0xE3;    // NACK for retrieve (e.g., not found)
const uint8_t SDSE_MSG_TYPE_DELETE_DATA_RESP = 0x84;      // ACK for delete if requested
const uint8_t SDSE_MSG_TYPE_ERROR_RESP = 0xFF;            // Generic error response from server

// Status Codes for Response Payloads (typically first byte of payload for RESP/NACK messages)
const uint8_t SDSE_STATUS_OK = 0x00;
const uint8_t SDSE_STATUS_ERROR_UNKNOWN = 0x01;         // Generic internal error
const uint8_t SDSE_STATUS_ERROR_ACCESS_DENIED = 0x02;   // ACL check failed (placeholder)
const uint8_t SDSE_STATUS_ERROR_NOT_FOUND = 0x03;       // Data object not found
const uint8_t SDSE_STATUS_ERROR_INVALID_REQUEST = 0x04; // Malformed request, bad payload etc.
const uint8_t SDSE_STATUS_ERROR_ALREADY_REGISTERED = 0x05; // For REGISTER_CLIENT if client ID (MRENCLAVE) already active
const uint8_t SDSE_STATUS_ERROR_NOT_REGISTERED = 0x06;   // Client performing operation without prior registration

// Structure to hold a parsed message from the client
struct SdseParsedMessage {
    uint8_t msg_type = 0;
    uint8_t flags = SDSE_FLAG_NONE;
    uint32_t payload_len = 0; 
    // Optional fields, presence determined by msg_type primarily
    std::vector<uint8_t> client_id; // For REGISTER_CLIENT_REQ, this is the MRENCLAVE hash
                                    // For other messages, it might be looked up via fd
    // uint32_t request_id = 0;     // Not used in this initial simple version
    std::vector<uint8_t> payload;   // Actual data payload

    // bool has_client_id = false;  // Implicitly known from msg_type or session
    // bool has_request_id = false; // Not used yet
};
