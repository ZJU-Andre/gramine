#include "sdse_client.h"
#include <iostream>
#include <vector>
#include <string>
#include <iomanip> // For std::hex, std::setw, std::setfill

// Helper to print vector<uint8_t> as hex or string
void print_data(const std::vector<uint8_t>& data, bool force_hex = false) {
    if (data.empty()) {
        std::cout << "<empty>";
        return;
    }
    bool is_printable = true;
    if (!force_hex) {
        for (uint8_t c : data) {
            if (!isprint(c) && !isspace(c)) {
                is_printable = false;
                break;
            }
        }
    } else {
        is_printable = false;
    }

    if (is_printable) {
        std::cout << "\"" << std::string(data.begin(), data.end()) << "\"";
    } else {
        std::cout << "0x";
        for (uint8_t byte : data) {
            std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(byte);
        }
        std::cout << std::dec;
    }
}

int main(int argc, char* argv[]) {
    std::string uds_path = "/tmp/sdse_socket.uds"; // Default UDS path

    if (argc > 1) {
        uds_path = argv[1];
        std::cout << "Using UDS path from command line: " << uds_path << std::endl;
    } else {
        std::cout << "Using default UDS path: " << uds_path << std::endl;
    }

    SdseClient client(uds_path);

    if (!client.connect_to_sdse()) {
        std::cerr << "Failed to connect to SDSE server." << std::endl;
        return 1;
    }

    // 1. Generate a dummy 32-byte client ID hash
    std::vector<uint8_t> client_id_hash(SDSE_CLIENT_ID_SIZE);
    for (size_t i = 0; i < SDSE_CLIENT_ID_SIZE; ++i) {
        client_id_hash[i] = static_cast<uint8_t>(0xA0 + i); // Example pattern
    }
    std::cout << "\nAttempting to register client with ID hash: ";
    print_data(client_id_hash, true);
    std::cout << std::endl;

    // 2. Register client
    if (client.register_client(client_id_hash)) {
        std::cout << "Client registration successful." << std::endl;
    } else {
        std::cerr << "Client registration failed." << std::endl;
        client.disconnect_from_sdse();
        return 1;
    }

    // 3. Store "obj1"
    std::string obj1_id = "obj1";
    std::vector<uint8_t> obj1_data = {'H', 'e', 'l', 'l', 'o', ',', ' ', 'S', 'D', 'S', 'E', '!'};
    std::cout << "\nAttempting to store data for '" << obj1_id << "': ";
    print_data(obj1_data);
    std::cout << std::endl;
    if (client.store_data(obj1_id, obj1_data)) {
        std::cout << "Store data for '" << obj1_id << "' succeeded." << std::endl;
    } else {
        std::cerr << "Store data for '" << obj1_id << "' failed." << std::endl;
    }

    // 4. Retrieve "obj1"
    std::vector<uint8_t> retrieved_data1;
    std::cout << "\nAttempting to retrieve data for '" << obj1_id << "'..." << std::endl;
    if (client.retrieve_data(obj1_id, retrieved_data1)) {
        std::cout << "Retrieved data for '" << obj1_id << "': ";
        print_data(retrieved_data1);
        std::cout << std::endl;
    } else {
        std::cerr << "Retrieve data for '" << obj1_id << "' failed." << std::endl;
    }

    // 5. Store "obj2"
    std::string obj2_id = "obj2_another_object";
    std::vector<uint8_t> obj2_data = {0x01, 0x02, 0x03, 0x04, 0x05, 0xDE, 0xAD, 0xBE, 0xEF};
    std::cout << "\nAttempting to store data for '" << obj2_id << "': ";
    print_data(obj2_data, true);
    std::cout << std::endl;
    if (client.store_data(obj2_id, obj2_data, false)) { // Store without requesting ACK
        std::cout << "Store data for '" << obj2_id << "' sent (no ACK requested)." << std::endl;
    } else {
        std::cerr << "Store data for '" << obj2_id << "' (no ACK) failed on send." << std::endl;
    }
    // Give server a moment to process if no ACK
    std::this_thread::sleep_for(std::chrono::milliseconds(100));


    // 6. Delete "obj1"
    std::cout << "\nAttempting to delete data for '" << obj1_id << "'..." << std::endl;
    if (client.delete_data(obj1_id)) {
        std::cout << "Delete data for '" << obj1_id << "' succeeded or object was not found (acknowledged)." << std::endl;
    } else {
        std::cerr << "Delete data for '" << obj1_id << "' failed." << std::endl;
    }
    
    // 7. Retrieve "obj1" again (should fail or return not found)
    std::vector<uint8_t> retrieved_data1_after_delete;
    std::cout << "\nAttempting to retrieve data for '" << obj1_id << "' after deletion..." << std::endl;
    if (client.retrieve_data(obj1_id, retrieved_data1_after_delete)) {
        std::cout << "Retrieved data for '" << obj1_id << "' (should have been deleted): ";
        print_data(retrieved_data1_after_delete);
        std::cout << std::endl;
    } else {
        std::cerr << "Retrieve data for '" << obj1_id << "' failed as expected (or error occurred)." << std::endl;
    }

    // 8. Retrieve "obj2" (should succeed)
    std::vector<uint8_t> retrieved_data2;
    std::cout << "\nAttempting to retrieve data for '" << obj2_id << "'..." << std::endl;
    if (client.retrieve_data(obj2_id, retrieved_data2)) {
        std::cout << "Retrieved data for '" << obj2_id << "': ";
        print_data(retrieved_data2, true);
        std::cout << std::endl;
    } else {
        std::cerr << "Retrieve data for '" << obj2_id << "' failed." << std::endl;
    }

    std::cout << "\nDisconnecting..." << std::endl;
    client.disconnect_from_sdse();

    return 0;
}
