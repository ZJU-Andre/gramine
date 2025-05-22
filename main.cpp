#include "server.h"
#include <iostream>
#include <string>
#include <csignal> // Required for signal handling
#include <atomic> // Required for std::atomic

// Global server instance pointer
Server* g_server_instance = nullptr;
// Global atomic flag to signal shutdown from main
extern std::atomic<bool> g_shutdown_flag; // Defined in server.cpp

// Signal handler function for main
void main_signal_handler(int signum) {
    std::cout << "Signal " << signum << " received in main, initiating server shutdown." << std::endl;
    g_shutdown_flag = true; // Set the global flag
    if (g_server_instance) {
        // g_server_instance->stop(); // This will be called by the server's own signal handler or when accept_connections loop ends
    }
}


int main(int argc, char* argv[]) {
    // Register signal handlers for SIGINT and SIGTERM in main
    signal(SIGINT, main_signal_handler);
    signal(SIGTERM, main_signal_handler);

    int port = 12345; // Default port
    if (argc > 1) {
        try {
            port = std::stoi(argv[1]);
        } catch (const std::invalid_argument& ia) {
            std::cerr << "Invalid port number: " << argv[1] << ". Using default port " << port << "." << std::endl;
        } catch (const std::out_of_range& oor) {
            std::cerr << "Port number out of range: " << argv[1] << ". Using default port " << port << "." << std::endl;
        }
    }

    // Define certificate and key paths
    // These files must exist for the server to start TLS.
    // Generate them using:
    // openssl req -x509 -newkey rsa:4096 -keyout server.key -out server.crt -days 365 -nodes
    const char* cert_path = "server.crt";
    const char* key_path = "server.key";

    Server server(port, cert_path, key_path);
    g_server_instance = &server; // Store address of server for signal handler

    server.start();

    // The server's accept_connections loop runs in a separate thread.
    // The main thread can wait here until a shutdown is signaled.
    // The server's internal signal handler or a call to stop() will cause accept_connections to return.
    while (!g_shutdown_flag && server.is_running()) { // server.is_running() is a hypothetical method, using g_shutdown_flag for now
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Check periodically
    }
    
    std::cout << "Shutdown signal received in main or server stopped, ensuring server cleanup." << std::endl;
    if (server.is_running()) { // Check if server is still marked as running
        server.stop(); // Ensure stop is called if not already initiated
    }
    
    std::cout << "Exiting main." << std::endl;
    return 0;
}

// Add is_running() method to Server class for main loop condition
// This requires modifying server.h and server.cpp
// server.h:
// public:
//    bool is_running() const { return running_; }
//
// server.cpp: (no changes needed for this, just the header)
//
// This change will be done in a separate step if needed.
// For now, the loop in main relies on g_shutdown_flag.
// Let's refine the server.cpp and server.h to ensure clean shutdown.
// The Server::stop() method should handle the server_socket_ correctly.
// The Server::accept_connections() loop should correctly break on g_shutdown_flag.
// The Server's signal handler should set g_shutdown_flag.

// Let's re-check server.cpp.
// The `g_shutdown_flag` is already used in `accept_connections`.
// The `signal_handler` in `server.cpp` sets `g_shutdown_flag`.
// The `main_signal_handler` also sets `g_shutdown_flag`.
// Server::stop() also sets g_shutdown_flag and running_ to false.

// The main loop `while (!g_shutdown_flag && server.is_running())` needs `server.is_running()`.
// I will add this method.
