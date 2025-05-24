#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <onnxruntime_c_api.h>
#include <cuda_runtime.h> // For cudaEvent_t

#define DEFAULT_MODEL_PATH "./models/mobilenetv2-7.onnx" // Relative to execution dir
#define ONNX_MODEL_INPUT_CHANNELS 3
#define ONNX_MODEL_INPUT_HEIGHT 224
#define ONNX_MODEL_INPUT_WIDTH 224
#define ONNX_MODEL_OUTPUT_CLASSES 1000

static const OrtApi* g_ort_api_native = NULL;
static OrtEnv* g_ort_env_native = NULL;
static OrtSession* g_ort_session_native = NULL;
static OrtAllocator* g_ort_allocator_native = NULL;
static const char* g_onnx_input_names_native[] = {"input"};
static const char* g_onnx_output_names_native[] = {"output"};

static int handle_ort_status_native(OrtStatus* status, const char* op_name) {
    if (status) {
        const char* msg = g_ort_api_native ? g_ort_api_native->GetErrorMessage(status) : "ONNX API not available";
        fprintf(stderr, "NATIVE_ONNX_ERROR: %s failed: %s\n", op_name, msg);
        if (g_ort_api_native) g_ort_api_native->ReleaseStatus(status);
        return 1;
    }
    return 0;
}

static int init_onnx_runtime_native(const char* model_path) {
    printf("NATIVE_ONNX_LOG: Initializing ONNX Runtime (model: %s)...\n", model_path);
    g_ort_api_native = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort_api_native) { fprintf(stderr, "NATIVE_ONNX_ERROR: Failed to get ONNX API base.\n"); return 1; }
    
    OrtStatus* status = g_ort_api_native->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "native-onnx-benchmark", &g_ort_env_native);
    if (handle_ort_status_native(status, "CreateEnv") != 0) return 1;

    OrtSessionOptions* session_options;
    status = g_ort_api_native->CreateSessionOptions(&session_options);
    if (handle_ort_status_native(status, "CreateSessionOptions") != 0) { goto err_cleanup_env; }

    // Enable CUDA Execution Provider
    printf("NATIVE_ONNX_LOG: Appending CUDA Execution Provider...\n");
    status = g_ort_api_native->SessionOptionsAppendExecutionProvider_CUDA_V2(session_options, NULL);
    if (handle_ort_status_native(status, "SessionOptionsAppendExecutionProvider_CUDA_V2") != 0) {
        fprintf(stderr, "NATIVE_ONNX_WARNING: Failed to append CUDA EP. Ensure ONNX Runtime was built with CUDA support.\n");
        // Continue, will try CPU or other available EPs
    } else {
        printf("NATIVE_ONNX_LOG: CUDA EP configured.\n");
    }

    status = g_ort_api_native->CreateSession(g_ort_env_native, model_path, session_options, &g_ort_session_native);
    g_ort_api_native->ReleaseSessionOptions(session_options);
    if (handle_ort_status_native(status, "CreateSession") != 0) { goto err_cleanup_env; }
    
    status = g_ort_api_native->GetAllocatorWithDefaultOptions(&g_ort_allocator_native);
    if (handle_ort_status_native(status, "GetAllocatorWithDefaultOptions") != 0) { goto err_cleanup_session; }

    printf("NATIVE_ONNX_LOG: ONNX Runtime initialized successfully.\n");
    return 0;

err_cleanup_session:
    if (g_ort_session_native) g_ort_api_native->ReleaseSession(g_ort_session_native);
    g_ort_session_native = NULL;
err_cleanup_env:
    if (g_ort_env_native) g_ort_api_native->ReleaseEnv(g_ort_env_native);
    g_ort_env_native = NULL;
    return 1;
}

static void shutdown_onnx_runtime_native() {
    printf("NATIVE_ONNX_LOG: Shutting down ONNX Runtime...\n");
    if (g_ort_api_native) {
        if (g_ort_session_native) g_ort_api_native->ReleaseSession(g_ort_session_native);
        // Allocator is default, no need to release explicitly
        if (g_ort_env_native) g_ort_api_native->ReleaseEnv(g_ort_env_native);
    }
}

static void generate_dummy_input(float* data, size_t num_elements) {
    for (size_t i = 0; i < num_elements; ++i) {
        data[i] = (float)rand() / RAND_MAX; // Values between 0 and 1
    }
}

int main(int argc, char* argv[]) {
    const char* model_path = DEFAULT_MODEL_PATH;
    if (argc > 1) {
        model_path = argv[1];
    }
    srand(time(NULL));

    if (init_onnx_runtime_native(model_path) != 0) {
        fprintf(stderr, "NATIVE_ONNX_ERROR: Failed to initialize ONNX runtime.\n");
        return 1;
    }

    const size_t input_elements = ONNX_MODEL_INPUT_CHANNELS * ONNX_MODEL_INPUT_HEIGHT * ONNX_MODEL_INPUT_WIDTH;
    const size_t input_size_bytes = input_elements * sizeof(float);
    float* input_tensor_values = (float*)malloc(input_size_bytes);
    if (!input_tensor_values) {
        fprintf(stderr, "NATIVE_ONNX_ERROR: Malloc failed for input tensor.\n");
        shutdown_onnx_runtime_native();
        return 1;
    }
    generate_dummy_input(input_tensor_values, input_elements);

    int64_t input_shape[] = {1, ONNX_MODEL_INPUT_CHANNELS, ONNX_MODEL_INPUT_HEIGHT, ONNX_MODEL_INPUT_WIDTH};
    OrtValue* input_tensor = NULL;
    OrtStatus* status = g_ort_api_native->CreateTensorWithDataAsOrtValue(
        g_ort_allocator_native, input_tensor_values, input_size_bytes,
        input_shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
    if (handle_ort_status_native(status, "CreateInputTensor") != 0) {
        free(input_tensor_values); shutdown_onnx_runtime_native(); return 1;
    }

    OrtValue* output_tensor = NULL;
    cudaEvent_t start_event, stop_event;
    float gpu_time_ms = 0.0f;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    printf("NATIVE_ONNX_LOG: Running inference...\n");
    cudaEventRecord(start_event, 0);
    status = g_ort_api_native->Run(g_ort_session_native, NULL, g_onnx_input_names_native,
                                 (const OrtValue* const*)&input_tensor, 1,
                                 g_onnx_output_names_native, 1, &output_tensor);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event); // Ensure GPU work is done for timing
    cudaEventElapsedTime(&gpu_time_ms, start_event, stop_event);
    
    if (handle_ort_status_native(status, "RunInference") != 0) {
        g_ort_api_native->ReleaseValue(input_tensor); free(input_tensor_values);
        shutdown_onnx_runtime_native(); cudaEventDestroy(start_event); cudaEventDestroy(stop_event);
        return 1;
    }
    printf("NATIVE_ONNX_LOG: Inference successful.\n");
    printf("NATIVE_ONNX_LOG: GPU execution time (OrtRun via CUDA EP): %.3f ms\n", gpu_time_ms);

    // Process output (optional, e.g., print top-k)
    const float* output_values;
    status = g_ort_api_native->GetTensorData(output_tensor, (const void**)&output_values);
    if (handle_ort_status_native(status, "GetOutputData") == 0) {
        // Basic check of output
        printf("NATIVE_ONNX_LOG: First output value: %f\n", output_values[0]);
    }

    g_ort_api_native->ReleaseValue(output_tensor);
    g_ort_api_native->ReleaseValue(input_tensor);
    free(input_tensor_values);
    shutdown_onnx_runtime_native();
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    printf("NATIVE_ONNX_LOG: Native ONNX benchmark finished.\n");
    return 0;
}
