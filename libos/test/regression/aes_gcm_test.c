#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "libos_aes_gcm.h"
#include "mbedtls/error.h" // For MBEDTLS_ERR_GCM_AUTH_FAILED
#include "mbedtls/platform_util.h" // For mbedtls_platform_zeroize

// Helper to print byte arrays for debugging
static void print_hex(const char* label, const unsigned char* data, size_t len) {
    printf("%s: ", label);
    for (size_t i = 0; i < len; ++i) {
        printf("%02x", data[i]);
    }
    printf("\n");
}

static int test_aes_gcm_encrypt_decrypt_success() {
    printf("Running test_aes_gcm_encrypt_decrypt_success...\n");

    unsigned char key[LIBOS_AES_GCM_KEY_SIZE_BYTES];
    unsigned char iv[LIBOS_AES_GCM_IV_SIZE_BYTES];
    unsigned char plaintext[] = "This is a test plaintext for AES-GCM.";
    size_t plaintext_len = sizeof(plaintext) - 1; // Exclude null terminator
    unsigned char ciphertext[sizeof(plaintext)]; // Same size as plaintext
    unsigned char decrypted_plaintext[sizeof(plaintext)];
    unsigned char tag[LIBOS_AES_GCM_TAG_SIZE_BYTES];
    unsigned char aad[] = "AdditionalAuthenticatedData";
    size_t aad_len = sizeof(aad) - 1;
    int ret;

    // Initialize key and IV (e.g., with dummy values for test)
    for (size_t i = 0; i < LIBOS_AES_GCM_KEY_SIZE_BYTES; ++i) key[i] = (unsigned char)(i + 1);
    for (size_t i = 0; i < LIBOS_AES_GCM_IV_SIZE_BYTES; ++i) iv[i] = (unsigned char)(i + 0x80);

    memset(ciphertext, 0, sizeof(ciphertext));
    memset(decrypted_plaintext, 0, sizeof(decrypted_plaintext));
    memset(tag, 0, sizeof(tag));

    // Encrypt
    ret = libos_aes_gcm_encrypt(key, iv, plaintext, plaintext_len, ciphertext, tag, aad, aad_len);
    assert(ret == 0 && "Encryption failed");
    if (ret != 0) {
        printf("Encryption failed with code: -0x%04X\n", (unsigned int)-ret);
        return -1;
    }
    // print_hex("Plaintext", plaintext, plaintext_len);
    // print_hex("Ciphertext", ciphertext, plaintext_len);
    // print_hex("Tag", tag, LIBOS_AES_GCM_TAG_SIZE_BYTES);
    // print_hex("AAD", aad, aad_len);


    // Decrypt
    ret = libos_aes_gcm_decrypt(key, iv, ciphertext, plaintext_len, tag, decrypted_plaintext, aad, aad_len);
    assert(ret == 0 && "Decryption failed");
     if (ret != 0) {
        printf("Decryption failed with code: -0x%04X\n", (unsigned int)-ret);
        return -1;
    }
    // print_hex("Decrypted Plaintext", decrypted_plaintext, plaintext_len);

    assert(memcmp(plaintext, decrypted_plaintext, plaintext_len) == 0 && "Decrypted plaintext does not match original");

    printf("test_aes_gcm_encrypt_decrypt_success: PASSED\n");
    return 0;
}

static int test_aes_gcm_decrypt_auth_failure() {
    printf("Running test_aes_gcm_decrypt_auth_failure...\n");

    unsigned char key[LIBOS_AES_GCM_KEY_SIZE_BYTES];
    unsigned char iv[LIBOS_AES_GCM_IV_SIZE_BYTES];
    unsigned char plaintext[] = "Another test for auth failure.";
    size_t plaintext_len = sizeof(plaintext) - 1;
    unsigned char ciphertext[sizeof(plaintext)];
    unsigned char decrypted_plaintext[sizeof(plaintext)];
    unsigned char tag[LIBOS_AES_GCM_TAG_SIZE_BYTES];
    unsigned char aad[] = "SomeAADContent";
    size_t aad_len = sizeof(aad) - 1;
    int ret;

    for (size_t i = 0; i < LIBOS_AES_GCM_KEY_SIZE_BYTES; ++i) key[i] = (unsigned char)(i + 0x1A);
    for (size_t i = 0; i < LIBOS_AES_GCM_IV_SIZE_BYTES; ++i) iv[i] = (unsigned char)(i + 0x9B);

    memset(ciphertext, 0, sizeof(ciphertext));
    memset(decrypted_plaintext, 0, sizeof(decrypted_plaintext));
    memset(tag, 0, sizeof(tag));

    // Encrypt first
    ret = libos_aes_gcm_encrypt(key, iv, plaintext, plaintext_len, ciphertext, tag, aad, aad_len);
    assert(ret == 0 && "Encryption step for auth failure test failed");
     if (ret != 0) {
        printf("Encryption step for auth failure test failed with code: -0x%04X\n", (unsigned int)-ret);
        return -1;
    }

    // Modify the tag (e.g., flip the first byte)
    tag[0] ^= 0xFF;
    // print_hex("Original Tag", tag_original, LIBOS_AES_GCM_TAG_SIZE_BYTES); // Need to copy tag before modifying
    // print_hex("Tampered Tag", tag, LIBOS_AES_GCM_TAG_SIZE_BYTES);


    // Attempt to decrypt with the modified tag
    ret = libos_aes_gcm_decrypt(key, iv, ciphertext, plaintext_len, tag, decrypted_plaintext, aad, aad_len);
    assert(ret == MBEDTLS_ERR_GCM_AUTH_FAILED && "Decryption did not report authentication failure as expected");
    if (ret != MBEDTLS_ERR_GCM_AUTH_FAILED) {
        printf("Decryption with tampered tag returned %d (expected %d / MBEDTLS_ERR_GCM_AUTH_FAILED -0x%04X)\n",
               ret, MBEDTLS_ERR_GCM_AUTH_FAILED, (unsigned int)-MBEDTLS_ERR_GCM_AUTH_FAILED);
        return -1;
    }


    printf("test_aes_gcm_decrypt_auth_failure: PASSED\n");
    return 0;
}

static int test_aes_gcm_encrypt_decrypt_various_lengths_single(size_t len, const unsigned char* aad_data, size_t aad_len) {
    printf("  Running for length %zu, AAD len %zu...\n", len, aad_len);
    unsigned char key[LIBOS_AES_GCM_KEY_SIZE_BYTES];
    unsigned char iv[LIBOS_AES_GCM_IV_SIZE_BYTES];
    unsigned char* plaintext = NULL;
    unsigned char* ciphertext = NULL;
    unsigned char* decrypted_plaintext = NULL;
    unsigned char tag[LIBOS_AES_GCM_TAG_SIZE_BYTES];
    int ret = -1; // Default to error

    plaintext = malloc(len > 0 ? len : 1); // Allocate 1 byte for 0-len to avoid NULL ptr issues with some mallocs
    ciphertext = malloc(len > 0 ? len : 1);
    decrypted_plaintext = malloc(len > 0 ? len : 1);

    if (!plaintext || !ciphertext || !decrypted_plaintext) {
        fprintf(stderr, "Malloc failed for length %zu\n", len);
        goto cleanup;
    }

    for (size_t i = 0; i < LIBOS_AES_GCM_KEY_SIZE_BYTES; ++i) key[i] = (unsigned char)(i + len); // Vary key slightly
    for (size_t i = 0; i < LIBOS_AES_GCM_IV_SIZE_BYTES; ++i) iv[i] = (unsigned char)(i + (len % 128)); // Vary IV
    for (size_t i = 0; i < len; ++i) plaintext[i] = (unsigned char)(i % 256);

    memset(ciphertext, 0, len > 0 ? len : 1);
    memset(decrypted_plaintext, 0, len > 0 ? len : 1);
    memset(tag, 0, sizeof(tag));

    ret = libos_aes_gcm_encrypt(key, iv, plaintext, len, ciphertext, tag, aad_data, aad_len);
    assert(ret == 0 && "Encryption failed (various_lengths)");
    if (ret != 0) {
        printf("Encryption (various_lengths) failed with code: -0x%04X for len %zu\n", (unsigned int)-ret, len);
        goto cleanup;
    }

    ret = libos_aes_gcm_decrypt(key, iv, ciphertext, len, tag, decrypted_plaintext, aad_data, aad_len);
    assert(ret == 0 && "Decryption failed (various_lengths)");
    if (ret != 0) {
        printf("Decryption (various_lengths) failed with code: -0x%04X for len %zu\n", (unsigned int)-ret, len);
        goto cleanup;
    }
    
    if (len > 0) { // memcmp with 0 length is tricky
        assert(memcmp(plaintext, decrypted_plaintext, len) == 0 && "Decrypted plaintext does not match original (various_lengths)");
    } else { // For 0-length, just ensure no error occurred
        assert(ret == 0);
    }

    ret = 0; // Success for this length

cleanup:
    free(plaintext);
    free(ciphertext);
    free(decrypted_plaintext);
    return ret;
}

static int test_aes_gcm_encrypt_decrypt_various_lengths() {
    printf("Running test_aes_gcm_encrypt_decrypt_various_lengths...\n");
    size_t lengths[] = {0, 1, 15, 16, 17, 32, 63, 64, 100, 1024};
    unsigned char aad1[] = "TestAAD1";
    unsigned char aad2[] = ""; // Empty AAD
    int final_ret = 0;

    for (size_t i = 0; i < sizeof(lengths) / sizeof(lengths[0]); ++i) {
        if (test_aes_gcm_encrypt_decrypt_various_lengths_single(lengths[i], aad1, sizeof(aad1)-1) != 0) {
            final_ret = -1;
        }
        if (test_aes_gcm_encrypt_decrypt_various_lengths_single(lengths[i], NULL, 0) != 0) { // Test with NULL AAD
            final_ret = -1;
        }
        if (test_aes_gcm_encrypt_decrypt_various_lengths_single(lengths[i], aad2, 0) != 0) { // Test with empty string AAD
            final_ret = -1;
        }
    }

    if (final_ret == 0) {
        printf("test_aes_gcm_encrypt_decrypt_various_lengths: PASSED\n");
    } else {
        printf("test_aes_gcm_encrypt_decrypt_various_lengths: FAILED for one or more lengths\n");
    }
    return final_ret;
}


int main() {
    int overall_ret = 0;

    if (test_aes_gcm_encrypt_decrypt_success() != 0) {
        overall_ret = -1;
    }
    if (test_aes_gcm_decrypt_auth_failure() != 0) {
        overall_ret = -1;
    }
    if (test_aes_gcm_encrypt_decrypt_various_lengths() != 0) {
        overall_ret = -1;
    }

    if (overall_ret == 0) {
        printf("All AES-GCM tests PASSED.\n");
        return 0;
    } else {
        printf("One or more AES-GCM tests FAILED.\n");
        return 1;
    }
}
