/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Borys Popławski <borysp@invisiblethingslab.com>
 */

/*
 * This file provides functions for dealing with outgoing IPC connections, mainly sending IPC
 * messages. In a shared-enclave model, a "client" enclave initiates connections to a "server"
 * (shared) enclave. This file primarily implements the client-side logic for initiating these
 * connections and sending messages. The server side (receiving connections and messages)
 * is handled elsewhere (e.g., in the main loop of the shared enclave's PAL (Platform Adaptation Layer)
 * or equivalent service listening mechanism).
 */

#include <stdbool.h>
#include <stdint.h>

#include "api.h"
#include "assert.h"
#include "avl_tree.h"
#include "libos_checkpoint.h"
#include "libos_internal.h"
#include "libos_ipc.h"
#include "libos_lock.h"
#include "libos_refcount.h"
#include "libos_types.h"
#include "libos_utils.h"
#include "pal.h"

struct libos_ipc_connection {
    struct avl_tree_node node;
    IDTYPE vmid; /* VMID of the remote enclave (e.g., the shared enclave this connection targets) */
    int seen_error; /* Last error encountered on this connection (e.g., -ECONNREFUSED, -EPIPE) */
    refcount_t ref_count;
    PAL_HANDLE handle; /* PAL handle for the IPC stream (pipe) */
    /* This lock guards concurrent accesses to `handle` and `seen_error`. If you need both this lock
     * and `g_ipc_connections_lock`, take the latter first. */
    struct libos_lock lock;
};

static bool ipc_connection_cmp(struct avl_tree_node* _a, struct avl_tree_node* _b) {
    struct libos_ipc_connection* a = container_of(_a, struct libos_ipc_connection, node);
    struct libos_ipc_connection* b = container_of(_b, struct libos_ipc_connection, node);
    return a->vmid <= b->vmid;
}

/* Tree of outgoing IPC connections (from this client enclave's perspective),
 * keyed by the destination vmid. Accessed only with `g_ipc_connections_lock` taken. */
static struct avl_tree g_ipc_connections = { .cmp = ipc_connection_cmp };
static struct libos_lock g_ipc_connections_lock;

struct ipc_msg_waiter {
    struct avl_tree_node node;
    PAL_HANDLE event;    /* Event to signal when a response is received or an error occurs */
    uint64_t seq;        /* Sequence number of the message sent, used to match with a response */
    IDTYPE dest;         /* VMID of the destination enclave for which this waiter is waiting */
    void* response_data; /* Buffer containing the response message, or NULL if error/no response */
};

static bool ipc_msg_waiter_cmp(struct avl_tree_node* _a, struct avl_tree_node* _b) {
    struct ipc_msg_waiter* a = container_of(_a, struct ipc_msg_waiter, node);
    struct ipc_msg_waiter* b = container_of(_b, struct ipc_msg_waiter, node);
    return a->seq <= b->seq;
}

static struct avl_tree g_msg_waiters_tree = { .cmp = ipc_msg_waiter_cmp };
static struct libos_lock g_msg_waiters_tree_lock;

struct libos_ipc_ids g_process_ipc_ids;

int init_ipc(void) {
    if (!create_lock(&g_ipc_connections_lock)) {
        return -ENOMEM;
    }
    if (!create_lock(&g_msg_waiters_tree_lock)) {
        return -ENOMEM;
    }

    return init_ipc_ids();
}

static void get_ipc_connection(struct libos_ipc_connection* conn) {
    refcount_inc(&conn->ref_count);
}

static void put_ipc_connection(struct libos_ipc_connection* conn) {
    refcount_t ref_count = refcount_dec(&conn->ref_count);

    if (!ref_count) {
        PalObjectDestroy(conn->handle);
        destroy_lock(&conn->lock);
        free(conn);
    }
}

static struct libos_ipc_connection* node2conn(struct avl_tree_node* node) {
    if (!node) {
        return NULL;
    }
    return container_of(node, struct libos_ipc_connection, node);
}

/*
 * Converts a destination VMID to a PAL pipe URI. This URI is used by client enclaves
 * to connect to a server (shared) enclave.
 * The format is "pipe:<host_instance_id>/<destination_vmid>".
 */
static int vmid_to_uri(IDTYPE vmid, char* uri, size_t uri_size) {
    int ret = snprintf(uri, uri_size, URI_PREFIX_PIPE "%lu/%u", g_pal_public_state->instance_id,
                       vmid);
    if (ret < 0 || (size_t)ret >= uri_size) {
        return -ERANGE;
    }

    return 0;
}

/*
 * Establishes an IPC connection to a destination enclave (e.g., a shared enclave).
 * If a connection to `dest` already exists in `g_ipc_connections`, its reference count is incremented.
 * Otherwise, a new connection is created:
 *  1. A pipe URI is generated using `vmid_to_uri`.
 *  2. `PalStreamOpen` is called to open the pipe. This is a blocking call on the client side
 *     that waits for the server enclave to accept the connection.
 *  3. The client enclave's own VMID (`g_process_ipc_ids.self_vmid`) is sent over the pipe to
 *     identify itself to the server. This initial handshake message is part of the protocol.
 *
 * Parameters:
 *  dest     - The VMID of the enclave to connect to (typically, the shared enclave's VMID).
 *  conn_ptr - Output parameter. On success, it will point to the (potentially new)
 *             connection object. The caller receives a new reference to this object
 *             and is responsible for calling `put_ipc_connection` when done.
 *
 * Returns:
 *  0 on success, or a negative errno value on failure.
 *  Possible errors include -ENOMEM (memory allocation), -ECONNREFUSED (if PalStreamOpen fails
 *  because the server is not listening or the pipe doesn't exist), or errors from `write_exact`.
 */
static int ipc_connect(IDTYPE dest, struct libos_ipc_connection** conn_ptr) {
    struct libos_ipc_connection dummy_key = { .vmid = dest };
    struct libos_ipc_connection* conn = NULL;
    int ret = 0;

    lock(&g_ipc_connections_lock);

    /* Check if a connection to this destination already exists */
    conn = node2conn(avl_tree_find(&g_ipc_connections, &dummy_key.node));
    if (conn) {
        get_ipc_connection(conn); /* Increment refcount for the caller */
        *conn_ptr = conn;
        unlock(&g_ipc_connections_lock);
        return 0; /* Found existing connection */
    }

    /* No existing connection, proceed to create a new one */
    unlock(&g_ipc_connections_lock); /* Unlock while creating, especially for PalStreamOpen */

    conn = calloc(1, sizeof(*conn));
    if (!conn) {
        return -ENOMEM;
    }

    if (!create_lock(&conn->lock)) {
        ret = -ENOMEM;
        goto err_free_conn;
    }

    conn->vmid = dest; /* Set vmid early for logging/debugging */
    refcount_set(&conn->ref_count, 1); /* Initial refcount for the connection itself (will be in tree) */

    char uri[PIPE_URI_SIZE];
    if (vmid_to_uri(dest, uri, sizeof(uri)) < 0) {
        log_error("IPC: Buffer for pipe URI to vmid %u is too small (PIPE_URI_SIZE %zu)",
                  dest, sizeof(uri));
        BUG(); /* This indicates a compile-time configuration error */
    }

    log_debug("IPC: Attempting to connect to vmid %u via URI '%s'", dest, uri);
    do {
        /* PAL_ACCESS_RDONLY seems counter-intuitive for a client sending a message,
         * but PAL pipes are typically bi-directional after connection.
         * The key is that the client *opens* the named pipe that the server *created*.
         * The server would have opened it in a mode that allows it to read (e.g. PAL_ACCESS_RDWR).
         * Let's assume this is correct as per PAL's pipe semantics.
         * If this were unidirectional, client would use WRONLY and server RDONLY.
         */
        ret = PalStreamOpen(uri, PAL_ACCESS_RDONLY, /*share_flags=*/0, PAL_CREATE_IGNORED,
                            /*options=*/0, &conn->handle);
    } while (ret == PAL_ERROR_INTERRUPTED);

    if (ret < 0) {
        log_warning("IPC: PalStreamOpen failed to connect to vmid %u (uri: %s): %s. "
                    "Server enclave might not be running or listening.",
                    dest, uri, pal_strerror(ret));
        ret = pal_to_unix_errno(ret);
        goto err_destroy_lock;
    }

    log_debug("IPC: Connection to vmid %u established (handle: %p). Sending self_vmid.", dest, conn->handle);
    /* Send our own VMID to the server to identify ourselves */
    ret = write_exact(conn->handle, &g_process_ipc_ids.self_vmid,
                      sizeof(g_process_ipc_ids.self_vmid));
    if (ret < 0) {
        log_error("IPC: Failed to send self_vmid to vmid %u after connect: %s",
                  dest, unix_strerror(ret));
        /* Connection was opened, but initial handshake failed.
         * Mark seen_error. The connection will be in the tree but subsequent sends will fail. */
        conn->seen_error = ret;
        /* We proceed to add it to the tree; cleanup will occur if sends fail or explicitly removed. */
    }

    /* Now, re-acquire lock to add the new connection to the global tree */
    lock(&g_ipc_connections_lock);
    /* Double check if another thread created the connection in the meantime */
    struct libos_ipc_connection* existing_conn_check = node2conn(avl_tree_find(&g_ipc_connections, &dummy_key.node));
    if (existing_conn_check) {
        /* Race condition: another thread created and inserted the connection.
         * Use the existing one, discard the one we just created. */
        unlock(&g_ipc_connections_lock);
        put_ipc_connection(conn); /* This will free our new conn if refcount hits 0 */

        get_ipc_connection(existing_conn_check);
        *conn_ptr = existing_conn_check;
        log_debug("IPC: Raced to create connection to vmid %u, using existing one.", dest);
        return 0;
    }

    /* Add the newly created connection to the tree */
    avl_tree_insert(&g_ipc_connections, &conn->node);
    unlock(&g_ipc_connections_lock);

    get_ipc_connection(conn); /* Increment refcount for the caller's reference */
    *conn_ptr = conn;
    log_debug("IPC: New connection to vmid %u (handle %p) successfully created and added to tree.",
              dest, conn->handle);
    return 0; /* Success */

err_destroy_lock:
    destroy_lock(&conn->lock);
err_free_conn:
    /* PalObjectDestroy for conn->handle is not needed here as it wasn't successfully opened or
       is handled by put_ipc_connection if it was partially successful before error.
       If seen_error was set after handle was opened, put_ipc_connection will close it.
       Refcount is 1, so put_ipc_connection will free it. */
    put_ipc_connection(conn);
    *conn_ptr = NULL;
    return ret;
}

/* Caller must hold g_ipc_connections_lock */
/* Caller must hold g_ipc_connections_lock */
static void _remove_ipc_connection(struct libos_ipc_connection* conn) {
    assert(locked(&g_ipc_connections_lock));
    avl_tree_delete(&g_ipc_connections, &conn->node);
    put_ipc_connection(conn); /* Decrements refcount, potentially freeing the connection and closing handle */
}

/*
 * A simple wrapper around ipc_connect to establish a connection without returning
 * the connection object itself. Useful for pre-establishing connections if needed,
 * or for cases where only the success/failure of the connection attempt matters immediately.
 */
int connect_to_process(IDTYPE dest) {
    struct libos_ipc_connection* conn = NULL;
    int ret = ipc_connect(dest, &conn);
    if (ret < 0) {
        return ret; /* ipc_connect would have logged the error */
    }
    /* ipc_connect returns a new reference (increments refcount). We must release it. */
    put_ipc_connection(conn);
    return 0;
}

/*
 * Removes an outgoing IPC connection to `dest`. This function is typically called
 * when the remote enclave (`dest`) is known to have terminated, is unresponsive,
 * or if this client enclave is shutting down its communication with that specific server.
 *
 * It performs two main actions:
 * 1. Removes the connection object from the global `g_ipc_connections` tree. This prevents
 *    new messages from being sent over this connection.
 * 2. Iterates through all pending message waiters (`g_msg_waiters_tree`) for this `dest`.
 *    For each such waiter, it sets their `response_data` to NULL (signaling an error or
 *    disconnection) and sets their event. This wakes up the threads that called
 *    `ipc_send_msg_and_get_response` so they don't hang indefinitely.
 */
void remove_outgoing_ipc_connection(IDTYPE dest) {
    struct libos_ipc_connection dummy_key = { .vmid = dest };
    struct libos_ipc_connection* conn_to_remove = NULL;

    lock(&g_ipc_connections_lock);
    conn_to_remove = node2conn(avl_tree_find(&g_ipc_connections, &dummy_key.node));
    if (conn_to_remove) {
        /* We need to be careful if _remove_ipc_connection itself could sleep or take other locks.
         * `_remove_ipc_connection` calls `put_ipc_connection` which calls `PalObjectDestroy`
         * and `destroy_lock`. These should be safe.
         */
        log_debug("IPC: Removing outgoing connection to vmid %u.", dest);
        _remove_ipc_connection(conn_to_remove);
    }
    unlock(&g_ipc_connections_lock);

    /* If a connection was removed (or even if not, a waiter might exist for a dest that
     * never successfully connected but for which a send was attempted),
     * wake up any threads waiting for a response from this destination.
     */
    lock(&g_msg_waiters_tree_lock);
    struct avl_tree_node* current_waiter_node = avl_tree_first(&g_msg_waiters_tree);
    while (current_waiter_node) {
        struct ipc_msg_waiter* waiter = container_of(current_waiter_node, struct ipc_msg_waiter, node);
        /* Advance to next node before potentially modifying/signaling the current waiter,
           as signaling might cause the waiter to be removed from the tree in another thread. */
        struct avl_tree_node* next_waiter_node = avl_tree_next(current_waiter_node);

        if (waiter->dest == dest) {
            log_debug("IPC: Connection to vmid %u removed or unavailable, notifying waiter for seq %lu.",
                      dest, waiter->seq);
            /* No data means error. The thread waiting in ipc_send_msg_and_get_response
             * will see response_data as NULL and handle it as an error. */
            waiter->response_data = NULL;
            PalEventSet(waiter->event);
            /* The waiter itself will be removed from g_msg_waiters_tree by
             * the ipc_send_msg_and_get_response function when it wakes up. */
        }
        current_waiter_node = next_waiter_node;
    }
    unlock(&g_msg_waiters_tree_lock);
}

void init_ipc_msg(struct libos_ipc_msg* msg, unsigned char code, size_t size) {
    SET_UNALIGNED(msg->header.size, size);
    SET_UNALIGNED(msg->header.seq, 0ul);
    SET_UNALIGNED(msg->header.code, code);
}

void init_ipc_response(struct libos_ipc_msg* msg, uint64_t seq, size_t size) {
    init_ipc_msg(msg, IPC_MSG_RESP, size);
    SET_UNALIGNED(msg->header.seq, seq);
}

/*
 * Sends an IPC message over an established and valid connection.
 * This function is an internal helper for `ipc_send_message` and `ipc_broadcast`.
 * It assumes `conn` is a valid, non-NULL pointer to an IPC connection object.
 *
 * Parameters:
 *  conn - The IPC connection to send the message on. This connection object must have
 *         been previously obtained via `ipc_connect` and should still be valid.
 *  msg  - The IPC message to send. The header (size, code, seq) must be initialized.
 *
 * Returns:
 *  0 on success.
 *  A negative errno value on failure. If `conn->seen_error` was already set (indicating
 *  a previous unrecoverable error on this connection), this error is returned immediately.
 *  If a new I/O error occurs during `write_exact`, `conn->seen_error` is updated with this
 *  new error, and the error is returned.
 */
static int ipc_send_message_to_conn(struct libos_ipc_connection* conn, struct libos_ipc_msg* msg) {
    log_debug("IPC: Sending message to vmid %u (handle %p): size %lu, code %u, seq %lu",
              conn->vmid, conn->handle, GET_UNALIGNED(msg->header.size),
              GET_UNALIGNED(msg->header.code), GET_UNALIGNED(msg->header.seq));

    int ret = 0;
    lock(&conn->lock); /* Protects access to conn->handle and conn->seen_error */

    if (conn->seen_error) {
        ret = conn->seen_error;
        log_debug("IPC: Connection to vmid %u (handle %p) has prior error %s; not sending new message.",
                  conn->vmid, conn->handle, unix_strerror(ret));
        goto out_unlock;
    }

    if (!conn->handle) { /* Should not happen if connection was successfully established and not yet closed */
        log_error("IPC: Cannot send message to vmid %u, connection handle is NULL.", conn->vmid);
        conn->seen_error = -ENOTCONN; /* Or -EBADF, treat as not connected */
        ret = conn->seen_error;
        goto out_unlock;
    }

    ret = write_exact(conn->handle, msg, GET_UNALIGNED(msg->header.size));
    if (ret < 0) {
        log_error("IPC: Failed to send message to vmid %u (handle %p): %s. Marking connection as broken.",
                  conn->vmid, conn->handle, unix_strerror(ret));
        conn->seen_error = ret; /* Mark connection as broken for future attempts on this conn object */
        /* Note: Higher-level logic might decide to call remove_outgoing_ipc_connection
         * based on this error, which would then clean up g_ipc_connections and notify waiters.
         * For a shared enclave, a client might retry by calling ipc_connect again to get a new conn.
         */
    }

out_unlock:
    unlock(&conn->lock);
    return ret;
}

/*
 * Sends an IPC message to a destination enclave.
 * This function will attempt to establish a connection using `ipc_connect` if one
 * doesn't already exist or if a previous attempt failed. If `ipc_connect` is successful,
 * it then uses `ipc_send_message_to_conn` to send the actual message.
 *
 * Parameters:
 *  dest - The VMID of the destination enclave (e.g., a shared enclave).
 *  msg  - The IPC message to send. The sequence number (`msg->header.seq`) should be 0
 *         if this is not a message expecting a response that will be handled by
 *         `ipc_send_msg_and_get_response` (which sets its own sequence numbers).
 *
 * Returns:
 *  0 on success, or a negative errno value on failure. This could be an error from
 *  connection establishment (`ipc_connect`) or from message sending (`ipc_send_message_to_conn`).
 */
int ipc_send_message(IDTYPE dest, struct libos_ipc_msg* msg) {
    struct libos_ipc_connection* conn = NULL;
    int ret = ipc_connect(dest, &conn); /* Obtains a connection with incremented refcount */
    if (ret < 0) {
        /* ipc_connect already logged the error (e.g., PalStreamOpen failure, ENOMEM) */
        log_warning("IPC: Failed to connect to vmid %u for sending message: %s",
                    dest, unix_strerror(ret));
        return ret;
    }

    /* Now conn is a valid pointer to a connection object with our reference on it. */
    ret = ipc_send_message_to_conn(conn, msg);
    put_ipc_connection(conn); /* Release our reference obtained from ipc_connect */
    return ret;
}

/*
 * Waits for a response to a previously sent IPC message.
 * This is a helper for `ipc_send_msg_and_get_response`.
 *
 * Parameters:
 *  waiter - The waiter object associated with the sent message. This object contains
 *           the event handle to wait on and other details like sequence number and destination.
 *
 * Returns:
 *  0 on success (event was signaled, indicating a response or error),
 *  or a negative errno value if `PalEventWait` itself fails (e.g., invalid handle).
 */
static int wait_for_response(struct ipc_msg_waiter* waiter) {
    log_debug("IPC: Waiting for response to seq %lu from dest %u (event %p)",
              waiter->seq, waiter->dest, waiter->event);

    int ret = 0;
    do {
        /* TODO: Consider adding a configurable timeout mechanism here, especially for
         * communication with shared enclaves where a client might not want to wait indefinitely.
         * This would require `PalEventWait` to support timeouts and then handling the timeout error.
         * For now, it waits indefinitely until PalEventSet is called or an unrecoverable error occurs.
         */
        ret = PalEventWait(waiter->event, /*timeout=*/NULL);
    } while (ret == PAL_ERROR_INTERRUPTED);

    if (ret < 0) {
        log_error("IPC: PalEventWait failed for seq %lu (dest %u, event %p): %s",
                  waiter->seq, waiter->dest, waiter->event, pal_strerror(ret));
    } else {
        log_debug("IPC: Wait for seq %lu (dest %u, event %p) finished; event signaled.",
                  waiter->seq, waiter->dest, waiter->event);
    }
    return pal_to_unix_errno(ret);
}

/*
 * Sends an IPC message and synchronously waits for a response.
 * This function handles the entire request-response cycle:
 *  1. Generates a unique sequence number for the message.
 *  2. Creates a waiter object (`ipc_msg_waiter`) to track the pending response.
 *     This includes creating a PAL event for synchronization.
 *  3. Adds the waiter to `g_msg_waiters_tree`.
 *  4. Sends the message using `ipc_send_message`.
 *  5. If sending is successful, waits for the waiter's event to be signaled using `wait_for_response`.
 *     This event is signaled by `ipc_response_callback` when a response arrives, or by
 *     `remove_outgoing_ipc_connection` if the connection is torn down.
 *  6. Once woken, it checks `waiter->response_data`:
 *     - If non-NULL, a response was received. Ownership of `response_data` is transferred to the caller
 *       via the `resp` output parameter (if `resp` is not NULL).
 *     - If NULL, it indicates an error (e.g., connection closed, remote enclave died, or explicit
 *       connection removal). An appropriate error code is determined (e.g., -ESRCH, or a specific
 *       error from `conn->seen_error`).
 *  7. Cleans up by removing the waiter from `g_msg_waiters_tree` and destroying the PAL event.
 *
 * Parameters:
 *  dest - The VMID of the destination enclave.
 *  msg  - The IPC message to send. Its `header.seq` will be populated by this function.
 *  resp - Output parameter. If the call is successful and a response is received, `*resp`
 *         will point to an allocated buffer containing the response data. The caller is
 *         responsible for `free()`ing this buffer. If an error occurs, or if the `resp`
 *         argument itself is NULL, `*resp` will be NULL (or unchanged if `resp` is NULL).
 *
 * Returns:
 *  0 on success (response received and *resp is set if resp was non-NULL).
 *  A negative errno value on failure. Possible errors include:
 *    -ENOMEM: Failed to allocate memory for waiter or event.
 *    -ESRCH:  The destination enclave terminated, the connection was explicitly removed,
 *             or some other issue prevented a response from being delivered (indicated by
 *             `waiter->response_data` being NULL after the wait).
 *    -ECONNREFUSED, -EHOSTUNREACH, etc.: Errors from underlying connection attempts.
 *    Other errors from `ipc_send_message` or `wait_for_response`.
 */
int ipc_send_msg_and_get_response(IDTYPE dest, struct libos_ipc_msg* msg, void** resp) {
    static uint64_t ipc_seq_counter = 1; /* Starts at 1; 0 is invalid/uninitialized for seq */
    uint64_t seq = __atomic_fetch_add(&ipc_seq_counter, 1, __ATOMIC_RELAXED);
    SET_UNALIGNED(msg->header.seq, seq);

    if (resp) {
        *resp = NULL; /* Initialize output parameter to known state */
    }

    struct ipc_msg_waiter waiter = {
        .node = {0}, /* Important to zero-initialize AVL node members like parent, left, right */
        .seq = seq,
        .dest = dest,
        .response_data = NULL, /* Will be populated by ipc_response_callback or error handling */
    };
    int ret = 0;

    /* Create an auto-clearing event. Once PalEventWait consumes the signal, it's reset. */
    ret = PalEventCreate(&waiter.event, /*init_signaled=*/false, /*auto_clear=*/true);
    if (ret < 0) {
        log_error("IPC: PalEventCreate failed for seq %lu: %s", seq, pal_strerror(ret));
        return pal_to_unix_errno(ret);
    }

    lock(&g_msg_waiters_tree_lock);
    if (!avl_tree_insert(&g_msg_waiters_tree, &waiter.node)) {
        /* This should ideally not happen if sequence numbers are unique and tree is managed correctly. */
        unlock(&g_msg_waiters_tree_lock);
        PalObjectDestroy(waiter.event);
        log_error("IPC: Failed to insert waiter for seq %lu into g_msg_waiters_tree (seq conflict?).", seq);
        return -EEXIST; /* Or -EINVAL; indicates an internal issue. */
    }
    unlock(&g_msg_waiters_tree_lock);

    log_debug("IPC: Sending message with seq %lu to dest %u, waiter %p created and added.",
              seq, dest, &waiter);
    ret = ipc_send_message(dest, msg);
    if (ret < 0) {
        log_warning("IPC: ipc_send_message failed for seq %lu to dest %u: %s. Waiter will be cleaned up.",
                    seq, dest, unix_strerror(ret));
        /* Message sending failed (e.g., connection error from ipc_connect or ipc_send_message_to_conn).
         * The waiter is in the tree, but wait_for_response will likely not be called,
         * or if called, it should ideally be unblocked by connection removal logic if applicable.
         * We must ensure the waiter is removed. */
        goto cleanup_waiter;
    }

    ret = wait_for_response(&waiter);
    if (ret < 0) {
        log_warning("IPC: wait_for_response failed for seq %lu from dest %u: %s. Waiter will be cleaned up.",
                    seq, dest, unix_strerror(ret));
        /* Waiting failed (e.g., PalEventWait error, not just a timeout if one was implemented).
         * This is less common than send errors or no response. */
        goto cleanup_waiter;
    }

    /* At this point, PalEventSet was called on waiter.event.
     * This could be due to:
     *   a) A valid response arriving (waiter.response_data is set by ipc_response_callback).
     *   b) Connection removal (waiter.response_data is set to NULL by remove_outgoing_ipc_connection).
     *   c) (Less common) Spurious wakeup if event wasn't auto-clear and was set by something else.
     *      (Using auto-clear events makes this less of a concern).
     */

    if (!waiter.response_data) {
        log_warning("IPC: No response data for seq %lu from dest %u (event %p signaled). "
                    "Connection likely closed, remote died, or explicitly removed.",
                    seq, dest, waiter.event);
        ret = -ESRCH; /* Default error for "process died" or "no such process/connection" */

        /* Check if the connection object itself has a more specific error stored.
         * This helps propagate errors like ECONNREFUSED from the initial connect attempt
         * or EPIPE from a later write failure on the connection.
         */
        struct libos_ipc_connection* conn_check = NULL;
        lock(&g_ipc_connections_lock);
        struct libos_ipc_connection dummy_key = { .vmid = dest };
        conn_check = node2conn(avl_tree_find(&g_ipc_connections, &dummy_key.node));
        if (conn_check) {
            get_ipc_connection(conn_check); /* Increment ref for safe access after unlock */
            unlock(&g_ipc_connections_lock);

            lock(&conn_check->lock);
            if (conn_check->seen_error) {
                log_debug("IPC: Overriding -ESRCH with connection's seen_error (%s) for seq %lu.",
                          unix_strerror(conn_check->seen_error), seq);
                ret = conn_check->seen_error;
            }
            unlock(&conn_check->lock);
            put_ipc_connection(conn_check); /* Release ref */
        } else {
            unlock(&g_ipc_connections_lock);
            /* If connection is not even in g_ipc_connections, it was likely removed. -ESRCH is appropriate. */
            log_debug("IPC: Connection to dest %u not found in g_ipc_connections for seq %lu; -ESRCH stands.",
                      dest, seq);
        }
    } else {
        /* Successfully received a response. waiter.response_data is valid. */
        if (resp) {
            *resp = waiter.response_data; /* Transfer ownership of buffer to caller */
            waiter.response_data = NULL;  /* Avoid double free during cleanup_waiter */
            log_debug("IPC: Response for seq %lu from dest %u received, data %p transferred to caller.",
                      seq, dest, *resp);
        } else {
            /* Caller didn't provide a `resp` pointer, so they don't want the data. We must free it. */
            log_debug("IPC: Response for seq %lu from dest %u received (data %p), but caller provided NULL resp. Freeing data.",
                      seq, dest, waiter.response_data);
            free(waiter.response_data);
            waiter.response_data = NULL;
        }
        ret = 0; /* Success */
    }

cleanup_waiter:
    lock(&g_msg_waiters_tree_lock);
    /* It's possible the waiter was already removed by some other path (e.g. if PalEventWait failed
     * catastrophically and we decided to remove it there, though current code doesn't do that).
     * avl_tree_delete should handle non-existent nodes gracefully (return NULL or no-op).
     */
    avl_tree_delete(&g_msg_waiters_tree, &waiter.node);
    unlock(&g_msg_waiters_tree_lock);

    /* Free response_data if it wasn't successfully transferred to the caller
     * (e.g., on an error path, or if *resp was NULL). */
    if (waiter.response_data) {
        log_debug("IPC: Cleaning up waiter: freeing potentially untransferred response_data %p for seq %lu.",
                  waiter.response_data, seq);
        free(waiter.response_data);
        waiter.response_data = NULL;
    }

    PalObjectDestroy(waiter.event);
    log_debug("IPC: Waiter for seq %lu (dest %u) fully cleaned up. Returning %s.",
              seq, dest, unix_strerror(ret));
    return ret;
}

/*
 * Callback function invoked by the underlying IPC message handling mechanism (e.g., PAL)
 * when an IPC message flagged as a "response" is received.
 *
 * Its primary jobs are:
 *  1. Validate the response (e.g., non-zero sequence number).
 *  2. Find the corresponding waiter object in `g_msg_waiters_tree` using the sequence number (`seq`).
 *  3. If a waiter is found:
 *     - Store the received `data` (which is the response payload) into `waiter->response_data`.
 *       Ownership of the `data` buffer is transferred to the waiter.
 *     - Signal the waiter's event (`waiter->event`) to wake up the thread that called
 *       `ipc_send_msg_and_get_response`.
 *  4. If no waiter is found, or if the response is invalid, the `data` buffer is freed.
 *
 * Parameters:
 *  src  - The VMID of the enclave that sent this response.
 *  data - Pointer to the allocated buffer containing the response message payload. This function
 *         takes ownership of this buffer if a matching waiter is found and the data is assigned.
 *         Otherwise, it frees the buffer.
 *  seq  - The sequence number of the response, used to match it with a sent message and its waiter.
 *
 * Returns:
 *  0 on success (waiter found and signaled).
 *  -EINVAL if `seq` is 0, no waiter is found for `seq`, or if the `src` VMID doesn't match
 *          the `dest` VMID recorded in the waiter (as a sanity check).
 *  If an error is returned, this function ensures `data` is freed.
 */
int ipc_response_callback(IDTYPE src, void* data, uint64_t seq) {
    int ret = 0;
    if (!seq) {
        log_error("IPC: Response received from vmid %u with invalid sequence number 0. Discarding.", src);
        ret = -EINVAL;
        goto out_free_data; /* Data must be freed as it won't be passed to any waiter */
    }

    lock(&g_msg_waiters_tree_lock);
    struct ipc_msg_waiter dummy_key_waiter = { .seq = seq };
    struct avl_tree_node* waiter_node = avl_tree_find(&g_msg_waiters_tree, &dummy_key_waiter.node);

    if (!waiter_node) {
        log_error("IPC: No waiter found for response with seq %lu from vmid %u. Late/unexpected response? Discarding.",
                  seq, src);
        ret = -EINVAL; /* No one is waiting for this specific response sequence. */
        goto out_unlock_free_data;
    }

    struct ipc_msg_waiter* waiter = container_of(waiter_node, struct ipc_msg_waiter, node);

    if (waiter->dest != src) {
        /* This is an important sanity check. The sequence number should be globally unique enough,
         * but if there's a bug or seq wrap-around, we might match a waiter for a different destination.
         * This check ensures the response came from the expected source. */
        log_warning("IPC: Response for seq %lu (waiter %p) expected from dest vmid %u, "
                    "but received from src vmid %u. Discarding response.",
                    seq, waiter, waiter->dest, src);
        ret = -EINVAL; /* Response source mismatch. */
        goto out_unlock_free_data;
    }

    if (waiter->response_data) {
        /* This implies a duplicate response for the same sequence number was received,
         * or there's a race where the waiter was signaled twice for the same data.
         * The first response usually wins. The current response_data is likely already being processed. */
        log_warning("IPC: Duplicate response or existing data for seq %lu (vmid %u, waiter %p). "
                    "Current data %p, new data %p. Discarding new response.",
                    seq, src, waiter, waiter->response_data, data);
        ret = -EALREADY; /* Or -EINVAL; indicates a protocol error or duplicate message. */
        goto out_unlock_free_data; /* Free the new data, keep the old one that's with the waiter. */
    }

    /* All checks passed. Transfer ownership of `data` to the waiter. */
    waiter->response_data = data;
    log_debug("IPC: Response from vmid %u for seq %lu (waiter %p) received, data %p assigned. Signaling waiter event %p.",
              src, seq, waiter, data, waiter->event);

    PalEventSet(waiter->event); /* Wake up the thread in ipc_send_msg_and_get_response */
    ret = 0; /* Success */

    unlock(&g_msg_waiters_tree_lock);
    return ret; /* `data` ownership is now with the waiter, so it's not freed here on success */

out_unlock_free_data:
    unlock(&g_msg_waiters_tree_lock);
out_free_data:
    if (data) {
        free(data); /* If we're not passing data to a waiter, free it here. */
    }
    return ret;
}

/*
 * Broadcasts an IPC message to all currently established outgoing connections,
 * except for the one whose `vmid` matches `exclude_id`.
 *
 * Note: This function iterates over *existing* connections in `g_ipc_connections`.
 * It does not attempt to discover or connect to all possible enclaves in the system.
 * It's suitable for scenarios where a client enclave needs to notify other enclaves
 * with which it has already established communication.
 *
 * Lock Ordering:
 * To avoid lock inversion, this function iterates `g_ipc_connections` under
 * `g_ipc_connections_lock`. For each connection, it increments its reference count,
 * releases `g_ipc_connections_lock`, sends the message (which involves taking `conn->lock`),
 * then re-acquires `g_ipc_connections_lock` to continue iteration.
 *
 * Parameters:
 *  msg        - The IPC message to broadcast.
 *  exclude_id - A VMID to exclude from the broadcast. Can be 0 or any other invalid ID
 *               if no exclusion is needed (though explicitly checking `conn->vmid != exclude_id`
 *               is the primary mechanism).
 *
 * Returns:
 *  0 if the message was sent successfully to all targeted connections (or if there were
 *    no targeted connections).
 *  If errors occur during sending, it returns the error code from the *first* failed
 *  `ipc_send_message_to_conn` encountered. Subsequent successful sends to other connections
 *  do not override this first error.
 */
int ipc_broadcast(struct libos_ipc_msg* msg, IDTYPE exclude_id) {
    int first_error = 0;
    struct avl_tree_node* current_node = NULL;

    lock(&g_ipc_connections_lock);
    current_node = avl_tree_first(&g_ipc_connections);

    while (current_node) {
        struct libos_ipc_connection* conn_iter = node2conn(current_node);
        /* Must get next node *before* releasing the lock and potentially modifying conn_iter or the tree.
           However, since we get a refcount on conn_iter, it won't be freed from under us.
           The main concern is current_node pointer validity across lock release/reacquire.
           So, get next node while lock is held.
        */
        struct avl_tree_node* next_node = avl_tree_next(&conn_iter->node);

        if (conn_iter->vmid == exclude_id) {
            current_node = next_node; /* Move to the next node in the tree */
            continue; /* Skip this connection */
        }

        get_ipc_connection(conn_iter); /* Increment refcount before releasing g_ipc_connections_lock */
        unlock(&g_ipc_connections_lock); /* Release lock before calling send, which takes conn_iter->lock */

        log_debug("IPC: Broadcasting message (seq %lu, code %u) to vmid %u (handle %p)",
                  GET_UNALIGNED(msg->header.seq), GET_UNALIGNED(msg->header.code),
                  conn_iter->vmid, conn_iter->handle);

        int ret = ipc_send_message_to_conn(conn_iter, msg);
        if (ret < 0 && first_error == 0) {
            log_warning("IPC: Broadcast to vmid %u failed with error: %s. Storing as first error.",
                        conn_iter->vmid, unix_strerror(ret));
            first_error = ret; /* Store the first error encountered */
        }

        put_ipc_connection(conn_iter); /* Decrement refcount for conn_iter */

        lock(&g_ipc_connections_lock); /* Re-acquire lock to continue iteration */
        current_node = next_node;      /* Advance to the previously determined next node */
    }

    unlock(&g_ipc_connections_lock);
    return first_error;
}

/*
 * Checkpointing support for IPC identifiers.
 * This function is part of Gramine's checkpoint/restore mechanism.
 * It saves the current enclave's IPC identifiers (`g_process_ipc_ids`)
 * into the checkpoint data.
 */
BEGIN_CP_FUNC(process_ipc_ids) {
    __UNUSED(size); /* Expected to be sizeof(struct libos_ipc_ids) */
    __UNUSED(objp); /* Not used if obj is the direct data */
    assert(size == sizeof(struct libos_ipc_ids));

    struct libos_ipc_ids* ids_to_checkpoint = (struct libos_ipc_ids*)obj;

    /* Calculate offset in the checkpoint area and add function entry for restoration */
    size_t offset_in_cp_area = ADD_CP_OFFSET(sizeof(*ids_to_checkpoint));
    ADD_CP_FUNC_ENTRY(offset_in_cp_area);

    /* Copy the current g_process_ipc_ids to the checkpoint location */
    *(struct libos_ipc_ids*)(base + offset_in_cp_area) = *ids_to_checkpoint;
    log_debug("IPC: Checkpointed process_ipc_ids (self_vmid: %u, parent_vmid: %u).",
              ids_to_checkpoint->self_vmid, ids_to_checkpoint->parent_vmid);
}
END_CP_FUNC(process_ipc_ids)

/*
 * Restoration support for IPC identifiers.
 * This function is part of Gramine's checkpoint/restore mechanism.
 * It restores the enclave's IPC identifiers from the checkpoint data
 * into the global `g_process_ipc_ids`.
 */
BEGIN_RS_FUNC(process_ipc_ids) {
    __UNUSED(offset); /* Offset within the checkpoint data, not directly used here */
    __UNUSED(rebase); /* Rebase structure, not directly used here */

    /* Retrieve the saved libos_ipc_ids from the checkpoint location */
    struct libos_ipc_ids* restored_ipc_ids = (void*)(base + GET_CP_FUNC_ENTRY());

    /* Restore the global g_process_ipc_ids */
    g_process_ipc_ids = *restored_ipc_ids;
    log_debug("IPC: Restored process_ipc_ids (self_vmid: %u, parent_vmid: %u).",
              g_process_ipc_ids.self_vmid, g_process_ipc_ids.parent_vmid);
}
END_RS_FUNC(process_ipc_ids)
