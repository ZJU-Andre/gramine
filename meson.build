#
# Project configuration, options, modules, scripts
#

project(
    'gramine',
    'c', 'cpp',
    version: '1.9post~UNRELEASED',
    license: 'LGPLv3+',

    meson_version: '>=0.58',

    default_options: [
        'c_std=c11',
        'cpp_std=c++14',
    ],
)

# If C23 (or experimetal C23 - C2x) is available, use it.
# TODO: Gramine supports older versions of gcc, so newer versions of compilers
#       may not be available.
#       We can change to c2x when we drop *EL8 and Ubuntu 20.04.
#       We can't change to c23 for any supported versions yet (requires at
#       least gcc 14).
if meson.get_compiler('c').has_argument('-std=c23')
  add_project_arguments('-std=c23', language: 'c')
elif meson.get_compiler('c').has_argument('-std=c2x')
  add_project_arguments('-std=c2x', language: 'c')
endif

# we need this subdir() early, because we need scripts defined there for setting up global vars
subdir('scripts')

pkglibdir = get_option('libdir') / meson.project_name()
pkgdatadir = get_option('datadir') / meson.project_name()

syslibdir = get_option('syslibdir')
if syslibdir == ''
    syslibdir = '/' / get_option('libdir')
endif

direct = get_option('direct') == 'enabled'
sgx = get_option('sgx') == 'enabled'
skeleton = get_option('skeleton') == 'enabled'

ubsan = get_option('ubsan') == 'enabled'
asan = get_option('asan') == 'enabled'
vtune = get_option('vtune') == 'enabled'

enable_libgomp = get_option('libgomp') == 'enabled'
enable_tests = get_option('tests') == 'enabled'

cc = meson.get_compiler('c')
host_has_glibc = cc.get_define('__GLIBC__', prefix: '#include <features.h>') != ''
objcopy = find_program('objcopy')

pythonmod = import('python')
python3 = pythonmod.find_installation('python3')
# TODO: after meson 0.60 (already in Debian, Ubuntu from jammy, not yet in RHEL), use
# -Dpython.platlibdir
python3_platlib = run_command(
    python3, get_python_platlib_prog, get_option('prefix')).stdout()
python3_pkgdir = python3_platlib / 'graminelibos'

pkgconfig = import('pkgconfig')

if host_machine.cpu_family() == 'x86_64'
    nasm_gen = generator(
        find_program('nasm'),
        output: '@BASENAME@.o',
        depfile: '@BASENAME@.dep',
        arguments: [
            '-f', 'elf64',
            '-p', '@SOURCE_DIR@/common/src/arch/x86_64/no_exec_stack.nasm',
            '-I', '@0@/'.format(meson.current_build_dir()),
            '-MQ', '@OUTPUT@', '-MF', '@DEPFILE@',
            '@EXTRA_ARGS@',
            '@INPUT@',
            '-o', '@OUTPUT@'
        ]
    )
endif

add_project_arguments(
    '-Wa,--noexecstack',

    '-Wall',
    '-Wextra',

    '-Wmissing-prototypes',
    '-Wstrict-prototypes',
    '-Wwrite-strings',

    cc.get_supported_arguments(
        '-Wtrampolines',
        '-Wnull-dereference',
    ),

    language: 'c')

debug = get_option('buildtype') == 'debug' or get_option('buildtype') == 'debugoptimized'
if debug
    add_project_arguments('-DDEBUG', language: 'c')
endif

if sgx
    cc.has_header('asm/sgx.h', required: true)

    if vtune
        add_project_arguments('-DSGX_VTUNE_PROFILE', language: 'c')
        vtune_sdk_path = get_option('vtune_sdk_path')
    endif
endif

gen_symbol_map_cmd = [
    find_program('nm'),
    '--numeric-sort', '--defined-only', '--print-size', '--line-numbers',
    '@INPUT@',
]

#
# Common checks and flags
#

# Not all compilers support mstack-protector-guard, so use stack protector only if supported.
# Gramine-custom stack protector uses the canary stored in the TCB (same for both in LibOS and PAL)
# at offset 0x8.
if host_machine.cpu_family() == 'x86_64'
    cflags_custom_stack_protector = [
        '-fstack-protector-strong',
        '-mstack-protector-guard=tls',
        '-mstack-protector-guard-reg=%gs',
        '-mstack-protector-guard-offset=8',
    ]
else
    cflags_custom_stack_protector = [
        '-fstack-protector-strong',
    ]
endif

if not cc.has_multi_arguments(cflags_custom_stack_protector)
    cflags_custom_stack_protector = '-fno-stack-protector'
endif

# Don't support b_sanitize: integration with sanitizers in Gramine is tricky and we want more
# control over the exact flags.
if get_option('b_sanitize') != 'none'
    error('Please don\'t use the b_sanitize option; use -Dubsan=enabled instead.')
endif

cflags_sanitizers = []

if ubsan
    cflags_sanitizers += [
        '-fsanitize=undefined',
        '-fno-sanitize-recover=undefined',
        '-DUBSAN',
    ]
endif

if asan
    if meson.get_compiler('c').get_id() != 'clang' or meson.get_compiler('cpp').get_id() != 'clang'
        error('ASan is currently supported only for Clang (CC=clang CXX=clang++)')
    endif

    asan_shadow_start = '0x18000000000'
    cflags_sanitizers += [
        '-fsanitize=address',
        '-fno-sanitize-link-runtime',
        '-mllvm', '-asan-mapping-offset=' + asan_shadow_start,
        '-DASAN',
    ]

    if meson.get_compiler('c').version().version_compare('>=13')
        cflags_sanitizers += ['-mllvm', '-asan-use-after-return=never']
    else
        cflags_sanitizers += ['-mllvm', '-asan-use-after-return=0']
    endif
endif

#
# Dependencies
#

tomlc99_proj = subproject('tomlc99-208203af46bdbdb29ba199660ed78d09c220b6c5')
tomlc99_dep = tomlc99_proj.get_variable('tomlc99_dep')
tomlc99_src = tomlc99_proj.get_variable('tomlc99_src')

uthash_dep = subproject('uthash-2.1.0').get_variable('uthash_dep')

mbedtls_proj = subproject('mbedtls-3.6.3')
mbedtls_static_dep = mbedtls_proj.get_variable('mbedtls_static_dep')
mbedtls_pal_dep = mbedtls_proj.get_variable('mbedtls_pal_dep')

curl_proj = subproject('curl-8.8.0')
cjson_proj = subproject('cJSON-1.7.12')
libcbor_proj = subproject('libcbor-0.11.0')

if sgx
    # XXX: do not call subproject() from under "if sgx" conditional, because it
    # breaks `meson dist`: `-Dsgx` checks for presence of sgx.h header and we
    # don't want to bother with finding the header just to create a tarball
    # TODO: this ^ probably can be revised when all distros will have the header
    # in <asm/sgx.h> so we won't have to mess with the paths anymore

    protoc_c_prog = find_program('protoc-c')

    threads_dep = dependency('threads')

    libcurl_dep = curl_proj.get_variable('curl_minimal_dep')

    cjson_dep = cjson_proj.get_variable('cjson_dep')

    protobuf_dep = dependency('libprotobuf-c')

    # NOTE: For libcbor build we use a custom_target (see subprojects/packagefiles/libcbor), because
    #       CMake integration in Meson doesn't work correctly with PIC static libraries, see
    #       https://github.com/mesonbuild/meson/issues/10764.
    libcbor_dep = libcbor_proj.get_variable('libcbor_dep')

    sgx_dcap_quoteverify_dep = cc.find_library('sgx_dcap_quoteverify',
        required: get_option('dcap'))

    vtune_dep = []
    dl_dep = []

    if vtune
        dl_dep = cc.find_library('dl', required: true)
        vtune_dep = declare_dependency(
            dependencies: [
                cc.find_library('ittnotify', dirs: vtune_sdk_path / 'lib64', required: true),
                dl_dep,
            ],
            include_directories: include_directories(vtune_sdk_path / 'include'),
        )
    endif
endif

#
# The compilation
#

subdir('common')
subdir('pal')
subdir('libos')
subdir('python')
subdir('tools')

if get_option('libc') == 'glibc'
    subproject('glibc-2.40-1')
elif get_option('libc') == 'musl'
    subproject('musl-1.2.5')
endif

if enable_libgomp
    subproject('gcc-10.2.0')
endif

run_target('clang-format', command: [meson_clang_format_prog])

conf = configuration_data()
conf.set('PATH', get_option('prefix') / get_option('bindir'))
conf.set('LD_LIBRARY_PATH', get_option('prefix') / get_option('libdir'))
conf.set('PKG_CONFIG_PATH', get_option('prefix') / get_option('libdir') / 'pkgconfig')
conf.set('PYTHONPATH', python3_platlib)
configure_file(
    input: 'scripts/activate.sh.in',
    output: 'activate.sh',
    configuration: conf,
)
