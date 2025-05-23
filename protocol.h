#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <cstdint>

// Message types for client-server communication
const uint8_t MSG_TYPE_REGISTER_CLIENT = 0x01;
const uint8_t MSG_TYPE_SEND_DATA = 0x02;
const uint8_t MSG_TYPE_POLL_DATA = 0x03;

// Response types from server
const uint8_t MSG_TYPE_REGISTER_ACK = 0x81;
const uint8_t MSG_TYPE_REGISTER_NACK = 0xE1; 
const uint8_t MSG_TYPE_SEND_ACK = 0x82;
const uint8_t MSG_TYPE_SEND_NACK = 0xE2;     
const uint8_t MSG_TYPE_DATA_AVAILABLE = 0x83;
const uint8_t MSG_TYPE_NO_DATA_AVAILABLE = 0xA3;
const uint8_t MSG_TYPE_POLL_NACK = 0xE3;     
const uint8_t MSG_TYPE_ERROR = 0xFF;         

// GPU Orchestrator specific message types
// Enclave to GPU Orchestrator (Commands)
const uint8_t GPU_ORCH_CMD_INIT_DEVICE = 0x20;
const uint8_t GPU_ORCH_CMD_ALLOC_MEM = 0x21;
const uint8_t GPU_ORCH_CMD_FREE_MEM = 0x22;
const uint8_t GPU_ORCH_CMD_COPY_H2D = 0x23; // Host (SHM) to Device
const uint8_t GPU_ORCH_CMD_COPY_D2H = 0x24; // Device to Host (SHM)
const uint8_t GPU_ORCH_CMD_LAUNCH_KERNEL = 0x25;
const uint8_t GPU_ORCH_CMD_SYNC_DEVICE = 0x26;
// Potentially: GPU_ORCH_CMD_LOAD_MODULE (0x27), GPU_ORCH_CMD_GET_KERNEL_FUNC (0x28)

// GPU Orchestrator to Enclave (Responses)
const uint8_t GPU_ORCH_RESP_ACK = 0xA0; // Generic ACK for simple commands
const uint8_t GPU_ORCH_RESP_NACK = 0xE0; // Generic NACK with error string payload
const uint8_t GPU_ORCH_RESP_ALLOC_MEM_SUCCESS = 0xA1; // Payload: gpu_ptr_handle (uint64_t)
// GPU_ORCH_RESP_DATA_READY was relevant for a different model. For D2H, data is in SHM, so ACK/NACK is sufficient.

#endif // PROTOCOL_H
