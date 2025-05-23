#include "orchestrator.h"
#include <iostream>
#include <string>
#include <csignal> // For signal handling

// External global shutdown flag, defined in orchestrator.cpp
extern std::atomic<bool> g_orchestrator_shutdown_flag;
// External signal handler function, defined in orchestrator.cpp
extern void orchestrator_signal_handler(int signum);

GpuOrchestrator* g_orchestrator_instance = nullptr; // For cleanup in custom signal handler if needed

void main_custom_signal_handler(int signum) {
    std::cout << "Main Orchestrator: Signal " << signum << " received." << std::endl;
    g_orchestrator_shutdown_flag = true; // Set the global flag
    if (g_orchestrator_instance) {
        // The server's internal signal handler should also trigger stop_server,
        // but this ensures it's called if the main loop is stuck or if signals are handled differently.
        // g_orchestrator_instance->stop_server(); // This might be problematic if called from signal handler directly
                                                // Best to rely on the flag and server's own handler.
    }
}


int main(int argc, char* argv[]) {
    // Register signal handlers for SIGINT and SIGTERM
    signal(SIGINT, main_custom_signal_handler);
    signal(SIGTERM, main_custom_signal_handler);
    // The GpuOrchestrator class could also register its own signal handler internally
    // if it needs more specific cleanup not covered by stop_server().


    int port = 12346; // Default port for orchestrator
    if (argc > 1) {
        try {
            port = std::stoi(argv[1]);
        } catch (const std::invalid_argument& ia) {
            std::cerr << "Invalid port number: " << argv[1] << ". Using default port " << port << "." << std::endl;
        } catch (const std::out_of_range& oor) {
            std::cerr << "Port number out of range: " << argv[1] << ". Using default port " << port << "." << std::endl;
        }
    }

    // TODO: Make cert_path and key_path configurable via command line arguments or config file
    const char* cert_path = "orchestrator.crt";
    const char* key_path = "orchestrator.key";
    std::cout << "Orchestrator will use certificate: " << cert_path << " and key: " << key_path << std::endl;
    std::cout << "Generate with: openssl req -x509 -newkey rsa:4096 -keyout orchestrator.key -out orchestrator.crt -days 365 -nodes -subj \"/CN=gpu-orchestrator\"" << std::endl;


    GpuOrchestrator orchestrator(port, cert_path, key_path);
    g_orchestrator_instance = &orchestrator;

    // Minimal CUDA initialization
    if (!orchestrator.init_cuda()) {
        std::cerr << "Failed to initialize CUDA. Exiting." << std::endl;
        return 1;
    }

    orchestrator.start_server();

    // Keep main thread alive while server runs in detached threads
    // The server's accept_connections loop will break when g_orchestrator_shutdown_flag is true.
    while (!g_orchestrator_shutdown_flag) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    
    std::cout << "Main Orchestrator: Shutdown signal received, ensuring server cleanup." << std::endl;
    // orchestrator.stop_server(); // stop_server() is now called by GpuOrchestrator destructor or its own signal handling,
                                // but an explicit call here ensures it if main loop exits first.
                                // The destructor of orchestrator will call stop_server() if it hasn't been called.
    // Forcing it here can be redundant if destructor handles it well.
    // Let's rely on the destructor for cleanup to avoid potential double-stop issues
    // if the signal handler in orchestrator.cpp also calls stop_server.

    std::cout << "Main Orchestrator: Exiting." << std::endl;
    return 0;
}
