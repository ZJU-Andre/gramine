#include "sdse_server.h"
#include <iostream>
#include <string>
#include <csignal>    // For signal, sigaction
#include <cstdlib>    // For exit, EXIT_FAILURE, EXIT_SUCCESS
#include <atomic>     // For std::atomic_bool (though server itself has one)

// Global pointer to the server instance for the signal handler
SdseServer* g_sdse_server_instance = nullptr;
// Global flag to indicate shutdown requested by signal
std::atomic<bool> g_sdse_shutdown_requested(false);

void sdse_signal_handler(int signum) {
    std::cout << "\nSDSE Main: Signal " << signum << " received. Initiating shutdown..." << std::endl;
    g_sdse_shutdown_requested = true;
    if (g_sdse_server_instance) {
        // This might be called from a signal handler context, so be careful.
        // The server's running_ flag will also be set by its stop() method.
        // Calling stop() here might be redundant if the main loop also checks the flag.
        // For now, we let the main loop handle the call to stop().
    }
}

int main(int argc, char* argv[]) {
    std::string uds_path = "/tmp/sdse_socket.uds"; // Default UDS path

    if (argc > 1) {
        uds_path = argv[1];
    }

    std::cout << "Starting SDSE Server on UDS path: " << uds_path << std::endl;

    // Setup signal handling for graceful shutdown
    struct sigaction sa;
    sa.sa_handler = sdse_signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0; // No SA_RESTART, so blocking calls like accept might return EINTR

    if (sigaction(SIGINT, &sa, nullptr) == -1) {
        perror("sigaction SIGINT failed");
        return EXIT_FAILURE;
    }
    if (sigaction(SIGTERM, &sa, nullptr) == -1) {
        perror("sigaction SIGTERM failed");
        return EXIT_FAILURE;
    }

    SdseServer server(uds_path);
    g_sdse_server_instance = &server; // Set global instance for signal handler (optional use)

    if (!server.start()) {
        std::cerr << "Failed to start SDSE server." << std::endl;
        return EXIT_FAILURE;
    }

    // Server's accept_loop runs in a separate thread.
    // Main thread can wait here until shutdown is signaled.
    while (!g_sdse_shutdown_requested) {
        // Keep main thread alive, check flag periodically.
        // The server's internal 'running_' flag controls its loops.
        // This g_sdse_shutdown_requested is primarily for the main function's loop.
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }

    std::cout << "SDSE Main: Shutdown requested. Stopping server..." << std::endl;
    server.stop(); // This will join the listener thread.
    
    std::cout << "SDSE Main: Server stopped. Exiting." << std::endl;
    g_sdse_server_instance = nullptr;
    return EXIT_SUCCESS;
}
