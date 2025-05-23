#ifndef LIBOS_AES_GCM_H
#define LIBOS_AES_GCM_H

#include <stddef.h> // For size_t

// AES-256 key size
#define LIBOS_AES_GCM_KEY_SIZE_BYTES (32)
// Recommended IV size for GCM
#define LIBOS_AES_GCM_IV_SIZE_BYTES (12)
// Standard GCM tag size
#define LIBOS_AES_GCM_TAG_SIZE_BYTES (16)

/*
 * TODO: Key Management:
 * The key used by these functions MUST be securely derived, managed, and protected
 * by the calling application or enclave. For SGX enclaves, consider using SGX sealing
 * mechanisms or deriving keys via remote attestation protocols to protect the AES key.
 * These functions assume the key is provided by a secure source.
 */

/*
 * TODO: IV (Initialization Vector) Management:
 * IVs MUST be unique for each encryption operation performed with the same key.
 * Reusing an IV with the same key completely undermines GCM's security.
 * It is recommended to generate IVs cryptographically randomly using a secure RNG.
 * The IV does not need to be secret, it can be transmitted with the ciphertext.
 */

/**
 * @brief Encrypts plaintext using AES-256-GCM.
 *
 * @param key The 256-bit (32-byte) encryption key.
 * @param iv The 96-bit (12-byte) initialization vector (IV). Must be unique for each encryption with the same key.
 * @param plaintext Pointer to the plaintext data to encrypt.
 * @param plaintext_len Length of the plaintext data in bytes.
 * @param ciphertext Pointer to the buffer where the ciphertext will be written.
 *                   This buffer must be at least `plaintext_len` bytes long.
 * @param tag Pointer to the buffer where the 128-bit (16-byte) authentication tag will be written.
 * @param aad Pointer to the additional authenticated data (AAD), or NULL if no AAD.
 * @param aad_len Length of the AAD in bytes. If aad is NULL, this should be 0.
 * @return 0 on success, or a negative mbed TLS error code on failure.
 */
int libos_aes_gcm_encrypt(
    const unsigned char key[LIBOS_AES_GCM_KEY_SIZE_BYTES],
    const unsigned char iv[LIBOS_AES_GCM_IV_SIZE_BYTES],
    const unsigned char *plaintext,
    size_t plaintext_len,
    unsigned char *ciphertext,
    unsigned char tag[LIBOS_AES_GCM_TAG_SIZE_BYTES],
    const unsigned char *aad,
    size_t aad_len
);

/**
 * @brief Decrypts ciphertext using AES-256-GCM and verifies the authentication tag.
 *
 * @param key The 256-bit (32-byte) decryption key.
 * @param iv The 96-bit (12-byte) initialization vector (IV) used during encryption.
 * @param ciphertext Pointer to the ciphertext data to decrypt.
 * @param ciphertext_len Length of the ciphertext data in bytes.
 * @param tag Pointer to the 128-bit (16-byte) authentication tag to verify.
 * @param plaintext Pointer to the buffer where the decrypted plaintext will be written.
 *                  This buffer must be at least `ciphertext_len` bytes long.
 * @param aad Pointer to the additional authenticated data (AAD) used during encryption, or NULL if no AAD.
 * @param aad_len Length of the AAD in bytes. If aad is NULL, this should be 0.
 * @return 0 on success (ciphertext decrypted and tag is valid).
 *         A negative mbed TLS error code on decryption failure or if the tag is invalid
 *         (e.g., MBEDTLS_ERR_GCM_AUTH_FAILED).
 */
int libos_aes_gcm_decrypt(
    const unsigned char key[LIBOS_AES_GCM_KEY_SIZE_BYTES],
    const unsigned char iv[LIBOS_AES_GCM_IV_SIZE_BYTES],
    const unsigned char *ciphertext,
    size_t ciphertext_len,
    const unsigned char tag[LIBOS_AES_GCM_TAG_SIZE_BYTES],
    unsigned char *plaintext,
    const unsigned char *aad,
    size_t aad_len
);

#endif /* LIBOS_AES_GCM_H */
