project('shared_service', 'c', 'cuda')

cc = meson.get_compiler('c')
nvcc = meson.get_compiler('cuda') # Will use `nvcc`

# CUDA runtime library. Prefer static linking if available and desired.
# For simplicity, this example might rely on Gramine's environment to find libcudart.so
# if not linking statically. For a robust build, one might specify paths or use find_library.
# This example will assume libcudart can be found by the linker (e.g. in system path or via LD_LIBRARY_PATH)
# For Gramine, often you'd want to package libcudart.so and other CUDA libs with the application.
# For now, let's just specify -lcudart.
cudart_dep = dependency('cudart', required: false) # Try to find via pkg-config or cmake
if not cudart_dep.found()
    # If not found by pkg-config/cmake, fallback to cc.find_library or just link with -lcudart
    cudart_dep = cc.find_library('cudart', required: true)
endif


# Adjust include paths based on Gramine's source structure relative to this example.
gramine_include_dir_libos = '../../../libos/include' 
gramine_include_dir_common_pal = '../../../common/include' # For PAL types if needed by libos_ipc.h
example_common_include_dir = '../../common' # For shared_service.h

# CUDA include directory (usually found by nvcc by default, but can be explicit)
# cuda_include_dir = '/usr/local/cuda/include' # Or from an env var

shared_service_inc = include_directories(
    'src', // Current source directory (for vector_add.h)
    gramine_include_dir_libos,
    gramine_include_dir_common_pal,
    example_common_include_dir
    # cuda_include_dir, # If needed explicitly
)

# Compile CUDA code (.cu) into an object file
# NVCC needs to know it's compiling for inclusion in C code (e.g. for name mangling of launch_vector_add_cuda)
# The `extern "C"` in the .cu file handles this.
# We also need to make it position-independent code if it's linked into a PIE executable or shared library.
# For a main enclave binary, this might not be strictly necessary but is good practice.
cuda_kernel_obj = nvcc.compile(
    'src/vector_add_kernel.cu',
    args: ['-Xcompiler=-fPIC', '--device-c'], // Compile for device, make position independent
    include_directories: shared_service_inc // nvcc also needs include paths
)


# Define the executable
executable('shared_service',
    'src/shared_service.c',
    cuda_kernel_obj, # Link with the compiled CUDA object
    include_directories: shared_service_inc,
    dependencies: [cudart_dep], # Link with CUDA runtime
    link_args: ['-Wl,-rpath=/usr/local/cuda/lib64'], # Example rpath, adjust as needed for your CUDA install
                                                 # Or ensure LD_LIBRARY_PATH is set in Gramine manifest/runtime.
    install: true,
    install_dir: 'bin' 
)

# Note on CUDA static linking:
# To link libcudart_static.a, you'd typically do something like:
# 1. Find libcudart_static.a, libculibos.a, etc.
# 2. Add them as objects or library dependencies.
# 3. Add linker flags like -Wl,--start-group obj1.o lib1.a lib2.a -Wl,--end-group
# This is more complex and platform-dependent. Dynamic linking with rpath or LD_LIBRARY_PATH
# is often simpler for examples. Gramine would need these .so files mounted.
#
# For the purpose of CI and this example, dynamic linking with -lcudart is assumed,
# and the runtime environment (e.g. Docker container for CI) is expected to have
# CUDA toolkit libraries available where the linker/loader can find them.
# The manifest would then need to mount these libraries.
# E.g. sgx.allowed_files = [ "file:/usr/local/cuda/lib64/libcudart.so.11.0" ] (adjust version)
# and possibly others like libcuda.so.1 (from driver).
# The `ld_library_path` in manifest can also be used: `loader.env.LD_LIBRARY_PATH = "/usr/local/cuda/lib64"`
#
# The current meson setup with `dependency('cudart')` or `cc.find_library('cudart')`
# usually resolves to dynamic linking. Static linking CUDA device code into an enclave that
# itself is a shared object (libsysdb.so loads the app) can be tricky.
# The provided setup for cuda_kernel_obj might need adjustment for device linking if not building a monolithic exe.
# However, for Gramine, shared_service is the main app, so this should be okay.
# The `--device-c` flag for nvcc is important for separate compilation of device code.
# If a full static executable is desired, nvcc link-time flags for static cudart would be needed.
# This build assumes `shared_service` is the main program for the enclave.The `CI-Examples/shared-enclave/meson.build` file was created in the previous turn.
This completes step 5 of the subtask.

**Summary of `shared_service.c` modifications and build setup:**

1.  **`CI-Examples/shared-enclave/src/shared_service.c` updated:**
    *   Included `shared_service.h` (common definitions), `vector_add.h` (for CUDA kernel wrapper), and `libos_aes_gcm.h` (for crypto).
    *   Defined a global, hardcoded AES key (`g_shared_enclave_aes_key`) for unmasking/masking data, with a TODO for secure key management.
    *   Renamed original `data_request_t`/`data_response_t` and their handlers to `legacy_...` to avoid conflicts and clarify their role.
    *   Implemented `handle_vector_add_request()`:
        *   Takes `vector_add_request_payload_t*` (from client) and `vector_add_response_payload_t*` (to fill for client).
        *   Validates `array_len_elements`.
        *   Allocates memory for plaintext float arrays (B_plain, C_plain, A_plain).
        *   Decrypts `masked_data_b` and `masked_data_c` from the request into `B_plain` and `C_plain` using `libos_aes_gcm_decrypt` with `g_shared_enclave_aes_key`.
        *   Calls `launch_vector_add_cuda(A_plain, B_plain, C_plain, ...)` to perform the GPU computation.
        *   Handles errors from decryption and CUDA launch, setting `resp_payload->status` appropriately.
        *   Generates a new IV for the result array A.
        *   Encrypts `A_plain` into `resp_payload->masked_data_a` (with tag `resp_payload->tag_a`) using `libos_aes_gcm_encrypt` and `g_shared_enclave_aes_key`.
        *   Sets `resp_payload->status = 0` on success and populates `resp_payload->array_len_elements`.
        *   Cleans up allocated plaintext memory.
    *   Modified `handle_client_session()`:
        *   Reads the `libos_ipc_msg_header_t` to get operation type and total expected message size.
        *   Reads the rest of the payload based on the header's size information.
        *   If `op_type == VECTOR_ADD_REQUEST`:
            *   Validates payload size against `sizeof(vector_add_request_payload_t)`.
            *   Calls `handle_vector_add_request()`.
            *   Prepares the response `libos_ipc_msg` using the populated `vector_add_response_payload_t`, setting the correct total size.
        *   If `op_type` is a legacy type, calls the legacy handlers.
        *   Sends the response message.
    *   The IPC send/receive helpers were simplified to `ipc_receive_raw_message_from_client` and `ipc_send_raw_response_to_client` which directly use `read_exact`/`write_exact` for the specified buffer length. This is because the `libos_ipc_msg` structure (header + data) is now being passed directly over the PAL pipe.

2.  **`CI-Examples/shared-enclave/meson.build` created:**
    *   Configures the build for the `shared_service` executable.
    *   Uses `nvcc` (via `meson.get_compiler('cuda')`) to compile `src/vector_add_kernel.cu` into an object file (`cuda_kernel_obj`) using `--device-c` and `-Xcompiler=-fPIC`.
    *   Compiles `src/shared_service.c` using the C compiler.
    *   Links `shared_service.c`'s object and `cuda_kernel_obj` together.
    *   Specifies include directories for Gramine LibOS headers (`../../../libos/include`, `../../../common/include`), local example common headers (`../../common`), and local source (`src`).
    *   Declares a dependency on `cudart` (CUDA runtime library), attempting to find it via `pkg-config` or `cc.find_library`.
    *   Includes an example `link_args` for `rpath` to help find CUDA libraries at runtime (this would typically be adapted to the specific deployment environment or handled by Gramine's manifest).
    *   Installs the resulting `shared_service` executable to a `bin` directory.
    *   Includes comments discussing CUDA static vs. dynamic linking considerations and how Gramine would typically manage CUDA library dependencies (e.g., via manifest `sgx.allowed_files` and `loader.env.LD_LIBRARY_PATH`).

This completes the subtask of updating the shared enclave service to handle the CUDA vector addition, including data unmasking/masking and build configuration.
