#include <iostream>
#include <tensorflow/c/c_api.h>

int main() {
    std::cout << "Hello from TensorFlow C library version " << TF_Version() << std::endl;

    // Print information about the TensorFlow runtime
    TF_Graph* graph = TF_NewGraph();
    TF_Status* status = TF_NewStatus();
    TF_SessionOptions* sessionOptions = TF_NewSessionOptions();
    TF_Session* session = TF_NewSession(graph, sessionOptions, status);

    if (TF_GetCode(status) != TF_OK) {
        std::cerr << "Failed to create TensorFlow session: " << TF_Message(status) << std::endl;
        TF_DeleteGraph(graph);
        TF_DeleteSessionOptions(sessionOptions);
        TF_DeleteStatus(status);
        return 1;
    }

    TF_DeviceList* devices = TF_SessionListDevices(session, status);
    int numDevices = TF_DeviceListCount(devices);
    std::cout << "Number of devices: " << numDevices << std::endl;

    for (int i = 0; i < numDevices; ++i) {
        const char* deviceName = TF_DeviceListName(devices, i, status);
        const char* deviceType = TF_DeviceListType(devices, i, status);
        std::cout << "Device " << i << ": " << deviceName << " (" << deviceType << ")" << std::endl;
    }

    TF_DeleteSession(session, status);
    TF_DeleteGraph(graph);
    TF_DeleteSessionOptions(sessionOptions);
    TF_DeleteStatus(status);

    return 0;
}
