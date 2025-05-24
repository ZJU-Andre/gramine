.PHONY: all client shared_service clean

MESON := meson
NINJA := ninja

BUILD_DIR_CLIENT := client-enclave/build
BUILD_DIR_SHARED := shared-enclave/build

all: client shared_service

client:
	@echo "Building client enclave..."
	@if [ ! -f "$(BUILD_DIR_CLIENT)/build.ninja" ]; then \
		$(MESON) setup $(BUILD_DIR_CLIENT) client-enclave --prefix=/usr/local; \
	fi
	$(NINJA) -C $(BUILD_DIR_CLIENT)
	@echo "Client enclave built."

shared_service:
	@echo "Building shared enclave service..."
	@if [ ! -f "$(BUILD_DIR_SHARED)/build.ninja" ]; then \
		$(MESON) setup $(BUILD_DIR_SHARED) shared-enclave --prefix=/usr/local; \
	fi
	$(NINJA) -C $(BUILD_DIR_SHARED)
	@echo "Shared enclave service built."

clean:
	@echo "Cleaning build directories..."
	rm -rf $(BUILD_DIR_CLIENT) $(BUILD_DIR_SHARED)
	@echo "Clean complete."

# Example of how signing might be added later:
# sign_client: client
# 	gramine-sgx-sign --manifest client-enclave/client-enclave.manifest --output client-enclave/client-enclave.manifest.sgx --key YOUR_KEY_FILE.pem
#
# sign_shared_service: shared_service
# 	gramine-sgx-sign --manifest shared-enclave/shared-enclave.manifest --output shared-enclave/shared-enclave.manifest.sgx --key YOUR_KEY_FILE.pem
#
# signed_all: sign_client sign_shared_service

# Note: The --prefix=/usr/local in meson setup is an example.
# The actual paths used in manifests (e.g., for sgx.trusted_files and libos.entrypoint)
# need to align with where these binaries are installed or how the Gramine environment
# is structured at runtime. For CI, this might involve copying built artifacts to specific
# locations that match the manifest paths.
# For instance, if entrypoint is "/gramine/CI-Examples/client-enclave/bin/client_app",
# the build system or a script would need to ensure client_app is placed there from Gramine's FS root perspective.
# The Makefile above installs to $(MESON_PREFIX)/bin by default from within the module build dir.
# This means client_app would be at /usr/local/bin/client_app with the current prefix.
# This discrepancy needs to be resolved at the CI/testing stage by either:
# 1. Adjusting manifest entrypoint paths to /usr/local/bin/client_app etc.
# 2. Using a different --prefix for meson setup (e.g., --prefix=/ for a chroot-like environment).
# 3. Staging files into a directory structure that Gramine will use as its root, matching the manifest.
# For this task, the Makefile provides build orchestration; path alignment is a runtime/deployment concern.
