#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <cstdint>

// Message types for client-server communication
const uint8_t MSG_TYPE_REGISTER_CLIENT = 0x01;
const uint8_t MSG_TYPE_SEND_DATA = 0x02;
const uint8_t MSG_TYPE_POLL_DATA = 0x03;

// Response types from server
const uint8_t MSG_TYPE_REGISTER_ACK = 0x81;
const uint8_t MSG_TYPE_REGISTER_NACK = 0xE1; // e.g. Client ID already in use by another active session (optional check)
const uint8_t MSG_TYPE_SEND_ACK = 0x82;
const uint8_t MSG_TYPE_SEND_NACK = 0xE2;     // e.g. Recipient not found or queue full
const uint8_t MSG_TYPE_DATA_AVAILABLE = 0x83;
const uint8_t MSG_TYPE_NO_DATA_AVAILABLE = 0xA3;
const uint8_t MSG_TYPE_POLL_NACK = 0xE3;     // e.g. Error during polling, or not registered
const uint8_t MSG_TYPE_ERROR = 0xFF;         // Generic error / unknown message

#endif // PROTOCOL_H
