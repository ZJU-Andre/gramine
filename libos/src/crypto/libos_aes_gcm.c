#include "libos_aes_gcm.h"
#include "mbedtls/gcm.h"
#include "mbedtls/aes.h" // Though gcm.h might include it, explicit include is good practice
#include "mbedtls/error.h" // For mbedtls_strerror
#include <string.h> // For memset, memcpy
#include <stdio.h>  // For debugging, remove later if not needed

// Helper to print mbedTLS errors - can be removed or ifdef'd for production
#if DEBUG_AES_GCM // Define this macro to enable error printing
static void print_mbedtls_error(const char* func_name, int err_code) {
    char error_buf[100];
    mbedtls_strerror(err_code, error_buf, sizeof(error_buf));
    fprintf(stderr, "LIBOS_AES_GCM: Error in %s: %s (0x%X)\n", func_name, error_buf, (unsigned int)-err_code);
}
#else
#define print_mbedtls_error(func_name, err_code) ((void)0)
#endif


int libos_aes_gcm_encrypt(
    const unsigned char key[LIBOS_AES_GCM_KEY_SIZE_BYTES],
    const unsigned char iv[LIBOS_AES_GCM_IV_SIZE_BYTES],
    const unsigned char *plaintext,
    size_t plaintext_len,
    unsigned char *ciphertext,
    unsigned char tag[LIBOS_AES_GCM_TAG_SIZE_BYTES],
    const unsigned char *aad,
    size_t aad_len
) {
    mbedtls_gcm_context ctx;
    int ret;

    if (!key || !iv || !plaintext || !ciphertext || !tag) {
        return MBEDTLS_ERR_GCM_BAD_INPUT; // Or a custom error like -EINVAL
    }
    if (plaintext_len > 0 && !plaintext) { // Check plaintext pointer if length > 0
        return MBEDTLS_ERR_GCM_BAD_INPUT;
    }
     if (aad_len > 0 && !aad) { // Check AAD pointer if length > 0
        return MBEDTLS_ERR_GCM_BAD_INPUT;
    }

    mbedtls_gcm_init(&ctx);

    ret = mbedtls_gcm_setkey(&ctx, MBEDTLS_CIPHER_ID_AES, key, LIBOS_AES_GCM_KEY_SIZE_BYTES * 8);
    if (ret != 0) {
        print_mbedtls_error("mbedtls_gcm_setkey (encrypt)", ret);
        mbedtls_gcm_free(&ctx);
        return ret;
    }

    // mbedtls_gcm_crypt_and_tag performs encryption and generates the tag
    ret = mbedtls_gcm_crypt_and_tag(
        &ctx,
        MBEDTLS_GCM_ENCRYPT,
        plaintext_len,
        iv,
        LIBOS_AES_GCM_IV_SIZE_BYTES,
        aad, // Additional Authenticated Data
        aad_len,
        plaintext,
        ciphertext,
        LIBOS_AES_GCM_TAG_SIZE_BYTES,
        tag
    );

    if (ret != 0) {
        print_mbedtls_error("mbedtls_gcm_crypt_and_tag", ret);
    }

    mbedtls_gcm_free(&ctx);
    return ret; // 0 on success, mbedTLS error code on failure
}

int libos_aes_gcm_decrypt(
    const unsigned char key[LIBOS_AES_GCM_KEY_SIZE_BYTES],
    const unsigned char iv[LIBOS_AES_GCM_IV_SIZE_BYTES],
    const unsigned char *ciphertext,
    size_t ciphertext_len,
    const unsigned char tag[LIBOS_AES_GCM_TAG_SIZE_BYTES],
    unsigned char *plaintext,
    const unsigned char *aad,
    size_t aad_len
) {
    mbedtls_gcm_context ctx;
    int ret;

    if (!key || !iv || !ciphertext || !tag || !plaintext) {
        return MBEDTLS_ERR_GCM_BAD_INPUT; // Or a custom error like -EINVAL
    }
     if (ciphertext_len > 0 && !ciphertext) { // Check ciphertext pointer if length > 0
        return MBEDTLS_ERR_GCM_BAD_INPUT;
    }
     if (aad_len > 0 && !aad) { // Check AAD pointer if length > 0
        return MBEDTLS_ERR_GCM_BAD_INPUT;
    }

    mbedtls_gcm_init(&ctx);

    ret = mbedtls_gcm_setkey(&ctx, MBEDTLS_CIPHER_ID_AES, key, LIBOS_AES_GCM_KEY_SIZE_BYTES * 8);
    if (ret != 0) {
        print_mbedtls_error("mbedtls_gcm_setkey (decrypt)", ret);
        mbedtls_gcm_free(&ctx);
        return ret;
    }

    // mbedtls_gcm_auth_decrypt performs decryption and verifies the tag
    ret = mbedtls_gcm_auth_decrypt(
        &ctx,
        ciphertext_len,
        iv,
        LIBOS_AES_GCM_IV_SIZE_BYTES,
        aad, // Additional Authenticated Data
        aad_len,
        tag,
        LIBOS_AES_GCM_TAG_SIZE_BYTES,
        ciphertext,
        plaintext
    );

    if (ret != 0) {
        // MBEDTLS_ERR_GCM_AUTH_FAILED is a common error if the tag doesn't match
        print_mbedtls_error("mbedtls_gcm_auth_decrypt", ret);
    }

    mbedtls_gcm_free(&ctx);
    return ret; // 0 on success, mbedTLS error code on failure (e.g., MBEDTLS_ERR_GCM_AUTH_FAILED)
}
