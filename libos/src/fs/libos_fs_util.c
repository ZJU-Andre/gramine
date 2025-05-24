/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Copyright (C) 2021 Intel Corporation
 *                    Paweł Marczewski <pawel@invisiblethingslab.com>
 */

#include "cpu.h"
#include "libos_flags_conv.h"
#include "libos_fs.h"
#include "libos_lock.h"
#include "libos_vma.h"
#include "stat.h"

int generic_seek(file_off_t pos, file_off_t size, file_off_t offset, int origin,
                 file_off_t* out_pos) {
    assert(pos >= 0);
    assert(size >= 0);

    switch (origin) {
        case SEEK_SET:
            pos = offset;
            break;

        case SEEK_CUR:
            if (__builtin_add_overflow(pos, offset, &pos))
                return -EOVERFLOW;
            break;

        case SEEK_END:
            if (__builtin_add_overflow(size, offset, &pos))
                return -EOVERFLOW;
            break;

        default:
            return -EINVAL;
    }

    if (pos < 0)
        return -EINVAL;

    *out_pos = pos;
    return 0;
}

int generic_readdir(struct libos_dentry* dent, readdir_callback_t callback, void* arg) {
    assert(locked(&g_dcache_lock));
    assert(dent->inode);
    assert(dent->inode->type == S_IFDIR);

    struct libos_dentry* child;
    LISTP_FOR_EACH_ENTRY(child, &dent->children, siblings) {
        if (child->inode) {
            int ret = callback(child->name, arg);
            if (ret < 0)
                return ret;
        }
    }
    return 0;
}

static int generic_istat(struct libos_inode* inode, struct stat* buf) {
    memset(buf, 0, sizeof(*buf));

    lock(&inode->lock);
    buf->st_mode = inode->type | inode->perm;
    buf->st_size = inode->size;
    buf->st_uid  = inode->uid;
    buf->st_gid  = inode->gid;
    buf->st_atime = inode->atime;
    buf->st_mtime = inode->mtime;
    buf->st_ctime = inode->ctime;

    /* Some programs (e.g. some tests from LTP) require this value. We've picked some random,
     * pretty looking constant - exact value should not affect anything (perhaps except
     * performance). */
    buf->st_blksize = 0x1000;
    /*
     * Pretend `nlink` is 2 for directories (to account for "." and ".."), 1 for other files.
     *
     * Applications are unlikely to depend on exact value of `nlink`, and for us, it's inconvenient
     * to keep track of the exact value (we would have to list the directory, and also take into
     * account synthetic files created by Gramine, such as named pipes and sockets).
     */
    buf->st_nlink = (inode->type == S_IFDIR ? 2 : 1);

    if (inode->mount->uri)
        buf->st_dev = hash_str(inode->mount->uri);

    unlock(&inode->lock);
    return 0;
}

int generic_inode_stat(struct libos_dentry* dent, struct stat* buf) {
    assert(locked(&g_dcache_lock));
    assert(dent->inode);

    return generic_istat(dent->inode, buf);
}

int generic_inode_hstat(struct libos_handle* hdl, struct stat* buf) {
    assert(hdl->inode);

    return generic_istat(hdl->inode, buf);
}

file_off_t generic_inode_seek(struct libos_handle* hdl, file_off_t offset, int origin) {
    file_off_t ret;

    if (!hdl->seekable)
        return 0;

    maybe_lock_pos_handle(hdl);
    lock(&hdl->inode->lock);
    file_off_t pos = hdl->pos;
    file_off_t size = hdl->inode->size;

    ret = generic_seek(pos, size, offset, origin, &pos);
    if (ret == 0) {
        hdl->pos = pos;
        ret = pos;
    }
    unlock(&hdl->inode->lock);
    maybe_unlock_pos_handle(hdl);
    return ret;
}

int generic_inode_poll(struct libos_handle* hdl, int in_events, int* out_events) {
    int ret;

    if (hdl->inode->type == S_IFREG) {
        ret = 0;
        *out_events = in_events & (POLLIN | POLLRDNORM | POLLOUT | POLLWRNORM);
    } else {
        ret = -EAGAIN;
    }

    return ret;
}

int generic_emulated_mmap(struct libos_handle* hdl, void* addr, size_t size, int prot, int flags,
                          uint64_t offset, size_t* out_valid_size) {
    assert(addr && IS_ALLOC_ALIGNED_PTR(addr));
    assert(IS_ALLOC_ALIGNED(size));
    assert(hdl && hdl->pal_handle); // Ensure pal_handle is valid

    int ret;
    pal_prot_flags_t pal_prot = LINUX_PROT_TO_PAL(prot, flags);

    // Check if the underlying PAL handle is a device
    if (hdl->pal_handle->hdr.type == PAL_TYPE_DEV) {
        log_debug("MMAP: Detected PAL_TYPE_DEV for fd %d, attempting PalDeviceMap addr %p size %zu off %lx prot %x",
                  hdl->fd, addr, size, offset, pal_prot);
        // For devices, directly call PalDeviceMap
        // Ensure that MAP_SHARED is passed if appropriate, as PalDeviceMap typically implies shared.
        // flags & MAP_SHARED is already incorporated into pal_prot by LINUX_PROT_TO_PAL
        ret = PalDeviceMap(hdl->pal_handle, addr, pal_prot, offset, size);
        if (ret < 0) {
            log_error("MMAP: PalDeviceMap failed for fd %d: %s", hdl->fd, pal_strerror(ret));
            return pal_to_unix_errno(ret);
        }
        // For device mappings, the entire requested size is considered valid if successful.
        // The PAL layer (and host OS) handles errors for invalid offset/size for a device.
        *out_valid_size = size;
        log_debug("MMAP: PalDeviceMap succeeded for fd %d, addr %p, size %zu", hdl->fd, addr, size);
        return 0;
    }

    // Original logic for regular files (emulated mmap)
    log_debug("MMAP: Not a device (type %d), proceeding with emulated mmap for fd %d", hdl->pal_handle->hdr.type, hdl->fd);
    pal_prot_flags_t pal_prot_writable = pal_prot | PAL_PROT_WRITE;

    ret = PalVirtualMemoryAlloc(addr, size, pal_prot_writable);
    if (ret < 0)
        return pal_to_unix_errno(ret);

    size_t size_to_read = size;
    char* read_addr = addr;
    file_off_t pos = offset;
    while (size_to_read > 0) {
        // For emulated mmap, we must use the fs_ops->read, not PalStreamRead directly,
        // as it might involve decryption or other fs-level logic.
        ssize_t count = hdl->fs->fs_ops->read(hdl, read_addr, size_to_read, &pos);
        if (count < 0) {
            if (count == -EINTR)
                continue;
            ret = count;
            goto err_emulated;
        }

        if (count == 0) // EOF
            break;

        assert((size_t)count <= size_to_read);
        size_to_read -= count;
        read_addr += count;
    }

    if (pal_prot != pal_prot_writable) {
        ret = PalVirtualMemoryProtect(addr, size, pal_prot);
        if (ret < 0) {
            ret = pal_to_unix_errno(ret);
            goto err_emulated;
        }
    }

    assert(size_to_read <= size);
    size_t valid_size = ALLOC_ALIGN_UP(size - size_to_read);
    if (valid_size < size) {
        // Protect pages that are beyond the end of the file
        int valid_ret = PalVirtualMemoryProtect(addr + valid_size, size - valid_size, PROT_NONE);
        if (valid_ret < 0) {
            log_error("MMAP: PalVirtualMemoryProtect for post-EOF region failed: %s", pal_strerror(valid_ret));
            // This is not necessarily a fatal error for the mapping itself, but indicates an issue.
            // Depending on strictness, could BUG() or just log. For now, log and continue.
        }
    }

    *out_valid_size = valid_size;
    log_debug("MMAP: Emulated mmap succeeded for fd %d, addr %p, valid_size %zu", hdl->fd, addr, valid_size);
    return 0;

err_emulated:;
    int free_ret = PalVirtualMemoryFree(addr, size);
    if (free_ret < 0) {
        log_debug("MMAP: PalVirtualMemoryFree failed on cleanup for emulated mmap (fd %d): %s", hdl->fd, pal_strerror(free_ret));
        // This is a serious issue, as we might leak memory or leave a VMA in an inconsistent state.
        BUG(); // Consider what to do in a production system if this happens.
    }
    return ret;
}

int generic_emulated_msync(struct libos_handle* hdl, void* addr, size_t size, int prot, int flags,
                           uint64_t offset) {
    assert(!(flags & MAP_PRIVATE));

    lock(&hdl->inode->lock);
    file_off_t file_size = hdl->inode->size;
    unlock(&hdl->inode->lock);

    pal_prot_flags_t pal_prot = LINUX_PROT_TO_PAL(prot, flags);
    pal_prot_flags_t pal_prot_readable = pal_prot | PAL_PROT_READ;

    int ret;
    if (pal_prot != pal_prot_readable) {
        ret = PalVirtualMemoryProtect(addr, size, pal_prot_readable);
        if (ret < 0)
            return pal_to_unix_errno(ret);
    }

    size_t write_size = offset > (uint64_t)file_size ? 0 : MIN(size, (uint64_t)file_size - offset);
    char* write_addr = addr;
    file_off_t pos = offset;
    while (write_size > 0) {
        ssize_t count = hdl->fs->fs_ops->write(hdl, write_addr, write_size, &pos);
        if (count < 0) {
            if (count == -EINTR)
                continue;
            ret = count;
            goto out;
        }

        if (count == 0) {
            log_debug("Failed to write back the whole mapping");
            ret = -EIO;
            goto out;
        }

        assert((size_t)count <= write_size);
        write_size -= count;
        write_addr += count;
    }

    ret = 0;

out:
    if (pal_prot != pal_prot_readable) {
        int protect_ret = PalVirtualMemoryProtect(addr, size, pal_prot);
        if (protect_ret < 0) {
            log_debug("PalVirtualMemoryProtect failed on cleanup: %s", pal_strerror(protect_ret));
            BUG();
        }
    }
    return ret;
}

int generic_truncate(struct libos_handle* hdl, file_off_t size) {
    lock(&hdl->inode->lock);
    int ret = PalStreamSetLength(hdl->pal_handle, size);
    if (ret < 0) {
        unlock(&hdl->inode->lock);
        return pal_to_unix_errno(ret);
    }

    hdl->inode->size = size;
    unlock(&hdl->inode->lock);

    refresh_mappings_on_file(hdl, size, /*reload_file_contents=*/false);
    return 0;
}
