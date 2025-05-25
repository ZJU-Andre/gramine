#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>    
#include <assert.h>
#include <cuda_runtime.h>

// Gramine includes
#include "libos_ipc.h"
#include "libos_aes_gcm.h"

// Shared header with service
#include "shared_service.h" // From CI-Examples/common/

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA_ERROR at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            /* Simplistic exit for POC */ \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Hardcoded session key for POC (ensure this matches a key the service might use for manifest decryption)
// This key is for encrypting the manifest itself.
static const unsigned char g_manifest_encryption_key[GCM_KEY_SIZE_BYTES] = {
    0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF,
    0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF, 0xDE, 0xAD, 0xBE, 0xEF
};

// Key for data segments (can be different from manifest key)
// This key is for encrypting/decrypting actual segment data when MASKING_AES_GCM is used.
static const unsigned char g_segment_data_key[GCM_KEY_SIZE_BYTES] = {
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f
};


// Helper to generate random IVs
static void generate_random_iv(unsigned char* iv, size_t iv_len) {
    for (size_t i = 0; i < iv_len; ++i) {
        iv[i] = (unsigned char)(rand() % 256);
    }
}

// POC function to encrypt the manifest
static int encrypt_manifest_for_poc(const workload_manifest_t* manifest, 
                                    const unsigned char* key, 
                                    batch_gpu_request_payload_t* request_payload) { // Using non-POC type
    if (!manifest || !key || !request_payload) {
        fprintf(stderr, "encrypt_manifest_for_poc: Invalid arguments.\n");
        return -1;
    }

    // Serialize the manifest (simple memcpy for POC)
    // In a real scenario, a proper serialization format like JSON or Protobuf would be used.
    unsigned char serialized_manifest_buffer[sizeof(workload_manifest_t)]; 
    if (sizeof(workload_manifest_t) > MAX_ENCRYPTED_MANIFEST_SIZE) { // Assuming MAX_ENCRYPTED_MANIFEST_SIZE
         fprintf(stderr, "encrypt_manifest_for_poc: sizeof(workload_manifest_t) %zu > MAX_ENCRYPTED_MANIFEST_SIZE %d\n",
            sizeof(workload_manifest_t), MAX_ENCRYPTED_MANIFEST_SIZE);
        return -1;
    }
    memcpy(serialized_manifest_buffer, manifest, sizeof(workload_manifest_t));
    uint32_t serialized_manifest_size = sizeof(workload_manifest_t);


    generate_random_iv(request_payload->encrypted_manifest_iv, GCM_IV_SIZE_BYTES);

    int ret = libos_aes_gcm_encrypt(key, 
                                    request_payload->encrypted_manifest_iv,
                                    serialized_manifest_buffer, 
                                    serialized_manifest_size,
                                    request_payload->encrypted_manifest_data, 
                                    request_payload->encrypted_manifest_tag,
                                    NULL, 0);
    if (ret != 0) {
        fprintf(stderr, "encrypt_manifest_for_poc: libos_aes_gcm_encrypt failed: %d\n", ret);
        return -1;
    }
    request_payload->encrypted_manifest_size = serialized_manifest_size;
    printf("CLIENT_APP_BATCH: Manifest encrypted successfully (size: %u).\n", serialized_manifest_size);
    return 0;
}


int main(int argc, char *argv[]) {
    printf("CLIENT_APP_BATCH_LOG: Starting Batch GPU Workload client application...\n");
    srand(time(NULL));
    int ret_status = EXIT_SUCCESS;

    // Buffers for segments
    unsigned char* aes_input_seg1_data_host = NULL;
    unsigned char* dma_input_seg2_buffer_host = NULL;
    uint64_t dma_input_seg2_buffer_device = 0;
    unsigned char* dma_output_seg3_buffer_host = NULL;
    unsigned char* aes_output_seg4_buffer_decrypted = NULL;

    struct libos_ipc_msg* response_msg_ptr = NULL;

    // 1. Hardcoded Workload Manifest (POC)
    workload_manifest_t client_manifest; 
    memset(&client_manifest, 0, sizeof(client_manifest));
    strncpy(client_manifest.manifest_id, "BatchClient_v1.0", MAX_MANIFEST_ID_LEN -1);
    client_manifest.num_segments = 4;
    assert(client_manifest.num_segments <= MAX_SEGMENTS_PER_MANIFEST);

    // Segment 1: "IMG_SEG" (High-Sensitivity Input AES-GCM)
    strncpy(client_manifest.segments[0].segment_id, "IMG_SEG", MAX_SEGMENT_ID_LEN -1);
    client_manifest.segments[0].sensitivity_level = SENSITIVITY_LEVEL_HIGH;
    client_manifest.segments[0].direction = DATA_DIRECTION_INPUT;
    strncpy(client_manifest.segments[0].gpu_operation_id, "OP_ENCRYPT_DECRYPT_TEST", MAX_GPU_OPERATION_ID_LEN-1);
    strncpy(client_manifest.segments[0].data_location_hint, "dummy_image_data_seg1", MAX_DATA_LOCATION_HINT_LEN-1);
    uint32_t size_seg1_data_plaintext = 64; 

    // Segment 2: "DMA_IN_SEG" (Low-Sensitivity DMA Input)
    strncpy(client_manifest.segments[1].segment_id, "DMA_IN_SEG", MAX_SEGMENT_ID_LEN -1);
    client_manifest.segments[1].sensitivity_level = SENSITIVITY_LEVEL_LOW;
    client_manifest.segments[1].direction = DATA_DIRECTION_INPUT;
    strncpy(client_manifest.segments[1].gpu_operation_id, "OP_DMA_PASSTHROUGH", MAX_GPU_OPERATION_ID_LEN-1);
    strncpy(client_manifest.segments[1].data_location_hint, "dummy_dma_input_data_seg2", MAX_DATA_LOCATION_HINT_LEN-1);
    uint32_t size_seg2_data_plaintext = 64;

    // Segment 3: "DMA_OUT_SEG" (Low-Sensitivity DMA Output)
    strncpy(client_manifest.segments[2].segment_id, "DMA_OUT_SEG", MAX_SEGMENT_ID_LEN -1);
    client_manifest.segments[2].sensitivity_level = SENSITIVITY_LEVEL_LOW;
    client_manifest.segments[2].direction = DATA_DIRECTION_OUTPUT;
    strncpy(client_manifest.segments[2].gpu_operation_id, "OP_DMA_PASSTHROUGH", MAX_GPU_OPERATION_ID_LEN-1);
    client_manifest.segments[2].client_dma_output_buffer_size = 128; 
    strncpy(client_manifest.segments[2].data_location_hint, "dummy_dma_output_location_seg3", MAX_DATA_LOCATION_HINT_LEN-1);

    // Segment 4: "AES_OUT_SEG" (High-Sensitivity Output AES-GCM)
    strncpy(client_manifest.segments[3].segment_id, "AES_OUT_SEG", MAX_SEGMENT_ID_LEN -1);
    client_manifest.segments[3].sensitivity_level = SENSITIVITY_LEVEL_HIGH;
    client_manifest.segments[3].direction = DATA_DIRECTION_OUTPUT;
    strncpy(client_manifest.segments[3].gpu_operation_id, "OP_ENCRYPT_DECRYPT_TEST", MAX_GPU_OPERATION_ID_LEN-1);
    strncpy(client_manifest.segments[3].data_location_hint, "dummy_aes_output_location_seg4", MAX_DATA_LOCATION_HINT_LEN-1);
    uint32_t max_expected_size_seg4_output = 64;


    // 2. Prepare Batch Request Payload
    batch_gpu_request_payload_t request_payload; 
    memset(&request_payload, 0, sizeof(request_payload));

    // 2.1 Encrypt Manifest for POC
    if (encrypt_manifest_for_poc(&client_manifest, g_manifest_encryption_key, &request_payload) != 0) {
        fprintf(stderr, "CLIENT_APP_BATCH_ERROR: Failed to encrypt manifest.\n");
        ret_status = EXIT_FAILURE; goto cleanup;
    }
    
    request_payload.num_segments = client_manifest.num_segments;
    assert(request_payload.num_segments <= MAX_SEGMENTS_PER_MANIFEST);

    // 2.2 Prepare Data and Segment Descriptors
    for (uint32_t i = 0; i < request_payload.num_segments; ++i) {
        workload_manifest_segment_t* manifest_seg = &client_manifest.segments[i];
        ipc_segment_descriptor_t* ipc_seg = &request_payload.segment_descriptors[i]; 

        strncpy(ipc_seg->segment_id, manifest_seg->segment_id, MAX_SEGMENT_ID_LEN -1);
        strncpy(ipc_seg->gpu_operation_id, manifest_seg->gpu_operation_id, MAX_GPU_OPERATION_ID_LEN -1);
        ipc_seg->masking_level = (manifest_seg->sensitivity_level == SENSITIVITY_LEVEL_HIGH) ? MASKING_AES_GCM : MASKING_NONE;
        
        if (strcmp(manifest_seg->segment_id, "IMG_SEG") == 0) {
            assert(size_seg1_data_plaintext <= MAX_INLINE_SEGMENT_DATA_SIZE);
            aes_input_seg1_data_host = (unsigned char*)malloc(size_seg1_data_plaintext);
            if (!aes_input_seg1_data_host) { ret_status = EXIT_FAILURE; goto cleanup; }
            for(uint32_t j=0; j<size_seg1_data_plaintext; ++j) aes_input_seg1_data_host[j] = (unsigned char)('A' + (j%26));
            
            generate_random_iv(ipc_seg->iv, GCM_IV_SIZE_BYTES); 
            if (libos_aes_gcm_encrypt(g_segment_data_key, ipc_seg->iv, aes_input_seg1_data_host, 
                                      size_seg1_data_plaintext,
                                      ipc_seg->encrypted_data, ipc_seg->tag, NULL, 0) != 0) {
                fprintf(stderr, "CLIENT_APP_BATCH_ERROR: Failed to encrypt IMG_SEG data.\n");
                ret_status = EXIT_FAILURE; goto cleanup;
            }
            ipc_seg->encrypted_data_size = size_seg1_data_plaintext;
            printf("CLIENT_APP_BATCH: Prepared IMG_SEG (AES-GCM Input, %u bytes to inline_encrypted_data).\n", ipc_seg->encrypted_data_size);

        } else if (strcmp(manifest_seg->segment_id, "DMA_IN_SEG") == 0) {
            CUDA_CHECK(cudaHostAlloc((void**)&dma_input_seg2_buffer_host, size_seg2_data_plaintext, cudaHostAllocDefault));
            for(uint32_t j=0; j<size_seg2_data_plaintext; ++j) dma_input_seg2_buffer_host[j] = (unsigned char)('0' + (j%10));
            CUDA_CHECK(cudaHostGetDevicePointer((void**)&dma_input_seg2_buffer_device, (void*)dma_input_seg2_buffer_host, 0));
            
            ipc_seg->src_device_ptr = dma_input_seg2_buffer_device;
            ipc_seg->input_data_size = size_seg2_data_plaintext; 
            printf("CLIENT_APP_BATCH: Prepared DMA_IN_SEG (DMA Input, ptr 0x%lx, size %u).\n", ipc_seg->src_device_ptr, ipc_seg->input_data_size);

        } else if (strcmp(manifest_seg->segment_id, "DMA_OUT_SEG") == 0) {
            CUDA_CHECK(cudaHostAlloc((void**)&dma_output_seg3_buffer_host, manifest_seg->client_dma_output_buffer_size, cudaHostAllocDefault));
            ipc_seg->dest_host_ptr = (uint64_t)dma_output_seg3_buffer_host;
            ipc_seg->dest_buffer_size_bytes = manifest_seg->client_dma_output_buffer_size;
            printf("CLIENT_APP_BATCH: Prepared DMA_OUT_SEG (DMA Output, host_ptr 0x%lx, buffer size %u).\n", ipc_seg->dest_host_ptr, ipc_seg->dest_buffer_size_bytes);
        
        } else if (strcmp(manifest_seg->segment_id, "AES_OUT_SEG") == 0) {
            aes_output_seg4_buffer_decrypted = (unsigned char*)malloc(max_expected_size_seg4_output);
            if (!aes_output_seg4_buffer_decrypted) { ret_status = EXIT_FAILURE; goto cleanup; }
            printf("CLIENT_APP_BATCH: Prepared AES_OUT_SEG (AES-GCM Output, client buffer for decryption, expected max size %u).\n", max_expected_size_seg4_output);
        }
    }

    // 3. Prepare IPC message
    struct libos_ipc_msg request_msg;
    init_ipc_msg(&request_msg, BATCH_GPU_REQUEST, 
                 sizeof(libos_ipc_msg_header_t) + sizeof(batch_gpu_request_payload_t));
    
    if (sizeof(batch_gpu_request_payload_t) > sizeof(request_msg.data)) {
         fprintf(stderr, "CLIENT_APP_BATCH_ERROR: batch_gpu_request_payload_t size (%zu) > libos_ipc_msg.data size (%zu)\n",
                 sizeof(batch_gpu_request_payload_t), sizeof(request_msg.data));
         ret_status = EXIT_FAILURE; goto cleanup;
    }
    memcpy(request_msg.data, &request_payload, sizeof(batch_gpu_request_payload_t));

    // 4. IPC Communication
    IDTYPE shared_enclave_vmid = 1; 
    printf("CLIENT_APP_BATCH_LOG: Sending BATCH_GPU_REQUEST to shared enclave (VMID: %u)...\n", shared_enclave_vmid);
    
    if (ipc_send_msg_and_get_response(shared_enclave_vmid, &request_msg, (void**)&response_msg_ptr) < 0) {
        fprintf(stderr, "CLIENT_APP_BATCH_ERROR: ipc_send_msg_and_get_response failed.\n");
        ret_status = EXIT_FAILURE; goto cleanup;
    }
    if (!response_msg_ptr) {
        fprintf(stderr, "CLIENT_APP_BATCH_ERROR: Received NULL response_msg_ptr from IPC.\n");
        ret_status = EXIT_FAILURE; goto cleanup;
    }
    printf("CLIENT_APP_BATCH_LOG: Received response from shared enclave.\n");

    // 5. Handle Response
    if (GET_UNALIGNED(response_msg_ptr->header.code) != BATCH_GPU_REQUEST) {
        fprintf(stderr, "CLIENT_APP_BATCH_ERROR: Received unexpected message code: %u\n", GET_UNALIGNED(response_msg_ptr->header.code));
        ret_status = EXIT_FAILURE; goto cleanup;
    }
    batch_gpu_response_payload_t* response_payload = (batch_gpu_response_payload_t*)response_msg_ptr->data; 
    if (response_payload->overall_batch_status != 0) {
        fprintf(stderr, "CLIENT_APP_BATCH_ERROR: Shared enclave reported overall_batch_status error: %d\n", response_payload->overall_batch_status);
    } else {
        printf("CLIENT_APP_BATCH_LOG: Overall batch status OK.\n");
    }

    printf("CLIENT_APP_BATCH_LOG: Processing %u segment responses...\n", response_payload->num_segments);
    for (uint32_t i = 0; i < response_payload->num_segments; ++i) {
        ipc_segment_response_t* seg_resp = &response_payload->segments[i]; 
        printf("  Segment ID: %s, Status: %d, Actual Output Size: %u\n", 
               seg_resp->segment_id, seg_resp->status, seg_resp->actual_output_data_size);

        if (seg_resp->status == 0) {
            if (strcmp(seg_resp->segment_id, "AES_OUT_SEG") == 0) {
                // Find the corresponding manifest segment to check sensitivity_level
                gpu_data_masking_level_t expected_output_masking = MASKING_NONE; // Default, should be updated
                for(uint32_t j=0; j < client_manifest.num_segments; ++j) {
                    if(strcmp(client_manifest.segments[j].segment_id, seg_resp->segment_id) == 0) {
                        expected_output_masking = (client_manifest.segments[j].sensitivity_level == SENSITIVITY_LEVEL_HIGH) ? MASKING_AES_GCM : MASKING_NONE;
                        break;
                    }
                }

                if (expected_output_masking == MASKING_AES_GCM && 
                    aes_output_seg4_buffer_decrypted && seg_resp->encrypted_data_size > 0 && 
                    seg_resp->encrypted_data_size <= MAX_INLINE_SEGMENT_DATA_SIZE) {
                    assert(seg_resp->encrypted_data_size <= max_expected_size_seg4_output);
                    if (libos_aes_gcm_decrypt(g_segment_data_key, seg_resp->iv, 
                                              seg_resp->encrypted_data, seg_resp->encrypted_data_size,
                                              seg_resp->tag, aes_output_seg4_buffer_decrypted, NULL, 0) == 0) {
                        printf("    AES_OUT_SEG decrypted data (first %u of %u bytes): ", 
                                seg_resp->encrypted_data_size > 16 ? 16 : seg_resp->encrypted_data_size,
                                seg_resp->encrypted_data_size);
                        for(uint32_t k=0; k < seg_resp->encrypted_data_size && k < 16; ++k) printf("%02x ", aes_output_seg4_buffer_decrypted[k]);
                        printf("\n");
                    } else {
                        fprintf(stderr, "    AES_OUT_SEG decryption failed!\n");
                    }
                }
            } else if (strcmp(seg_resp->segment_id, "DMA_OUT_SEG") == 0) {
                 gpu_data_masking_level_t expected_output_masking = MASKING_NONE;
                 uint32_t expected_buffer_size = 0;
                 for(uint32_t j=0; j < client_manifest.num_segments; ++j) {
                    if(strcmp(client_manifest.segments[j].segment_id, seg_resp->segment_id) == 0) {
                        expected_output_masking = (client_manifest.segments[j].sensitivity_level == SENSITIVITY_LEVEL_HIGH) ? MASKING_AES_GCM : MASKING_NONE;
                        expected_buffer_size = client_manifest.segments[j].client_dma_output_buffer_size;
                        break;
                    }
                }
                if (expected_output_masking == MASKING_NONE &&
                    dma_output_seg3_buffer_host && seg_resp->actual_output_data_size > 0) {
                    assert(seg_resp->actual_output_data_size <= expected_buffer_size);
                    printf("    DMA_OUT_SEG data in client buffer (first %u of %u bytes): ", 
                           seg_resp->actual_output_data_size > 16 ? 16 : seg_resp->actual_output_data_size,
                           seg_resp->actual_output_data_size);
                    for(uint32_t k=0; k < seg_resp->actual_output_data_size && k < 16; ++k) printf("%02x ", dma_output_seg3_buffer_host[k]);
                    printf("\n");
                }
            }
        }
    }

cleanup:
    printf("CLIENT_APP_BATCH_LOG: Cleaning up resources...\n");
    if (response_msg_ptr) free(response_msg_ptr);
    if (aes_input_seg1_data_host) free(aes_input_seg1_data_host);
    if (aes_output_seg4_buffer_decrypted) free(aes_output_seg4_buffer_decrypted);
    if (dma_input_seg2_buffer_host) CUDA_CHECK(cudaFreeHost(dma_input_seg2_buffer_host));
    if (dma_output_seg3_buffer_host) CUDA_CHECK(cudaFreeHost(dma_output_seg3_buffer_host));
    
    if (ret_status == EXIT_SUCCESS) {
        printf("CLIENT_APP_BATCH_LOG: Batch GPU Workload client finished successfully.\n");
    } else {
        printf("CLIENT_APP_BATCH_LOG: Batch GPU Workload client finished with errors.\n");
    }
    return ret_status;
}
