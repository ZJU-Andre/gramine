project('client_app', 'c')

cc = meson.get_compiler('c')

# Find Gramine LibOS dependency (libsysdb.so)
# This assumes Gramine has been installed and pkg-config files are available.
# If Gramine is built as a subproject, the dependency might be obtained differently.
# For CI, we assume Gramine's libos (sysdb) is findable via environment or rpath.
# In a typical Gramine setup, applications are linked against a loader (gramine-direct/gramine-sgx)
# which then loads libsysdb.so. For direct compilation of an app that uses libos headers/libs,
# you'd need to ensure linkage.
# However, for this example, we are building a source file that will be the entrypoint
# of a Gramine LibOS instance. It needs access to libos_ipc.h and libos_aes_gcm.h.
# These headers are part of the Gramine source tree (libos/include).
# The actual linking against mbedTLS for AES-GCM and the IPC mechanisms happens when libsysdb.so itself is built.
# This client_app.c will be part of the sources compiled *into* the Gramine manifest's entrypoint.

# Therefore, we don't link client_app against libsysdb.so here.
# Instead, client_app.c is compiled into an executable that the manifest will run *inside* Gramine.
# It needs include paths to Gramine's headers.

# Adjust include paths based on Gramine's source structure relative to this example.
# Assuming this CI-Examples directory is at the root of the Gramine repo, or
# Gramine's headers are installed system-wide or found via an environment variable.

gramine_include_dir_libos = '../../libos/include' 
gramine_include_dir_common = '../../common/include' # For PAL types if needed by libos_ipc.h
example_common_include_dir = '../common' # For shared_service.h

client_app_inc = include_directories(
    '.', // Current source directory
    gramine_include_dir_libos,
    gramine_include_dir_common,
    example_common_include_dir
)

executable('client_app',
    'src/client_app.c',
    include_directories: client_app_inc,
    install: true,
    install_dir: 'bin' 
)

executable('client_app_onnx',
    'src/client_app_onnx.c',
    include_directories: client_app_inc,
    install: true,
    install_dir: 'bin'
)

executable('client_app_gemm',
    'src/client_app_gemm.c',
    include_directories: client_app_inc,
    install: true,
    install_dir: 'bin'
)

# Note: The manifest for this client_app will then specify:
# sgx.trusted_files = [ "file:bin/client_app" ]
# libos.entrypoint = "/bin/client_app" (or similar, matching install_dir)
#
# The client_app executable itself doesn't need to link mbedTLS or IPC internals,
# as those are provided by the LibOS environment (libsysdb.so) it runs within.
# The key is that client_app.c can successfully include the necessary headers.
# If `libos_ipc.c` or `libos_aes_gcm.c` were compiled as static libraries and linked
# here, then this executable would be much larger and self-contained, but that's
# not the typical Gramine model for the main application.
