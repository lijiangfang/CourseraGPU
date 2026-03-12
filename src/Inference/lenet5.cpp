#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>
#include <cudnn.h>
#include <cublas_v2.h>
#include "cucheck.h"
#include "lenet5.h"
#include "bias_relu_kernel.h"

namespace fs = std::filesystem;

LeNet5::LeNet5()
{
    d_conv1_w = nullptr;
    d_conv1_b = nullptr;
    d_conv2_w = nullptr;
    d_conv2_b = nullptr;
    d_fc1_w = nullptr;
    d_fc1_b = nullptr;
    d_fc2_w = nullptr;
    d_fc2_b = nullptr;
    d_fc3_w = nullptr;
    d_fc3_b = nullptr;

    d_pool1_out = nullptr;
    d_conv2_out = nullptr;
    d_pool2_out = nullptr;
    d_fc1_out = nullptr;
    d_fc2_out = nullptr;
    workspace = nullptr;

    cudnn = nullptr;
    cublas = nullptr;
}

LeNet5::~LeNet5()
{
    cleanup();
}

void LeNet5::setup(std::string &modelPath, int max_batch_size)
{
    // load the model weights from binary files into GPU memory
    loadModel(modelPath);

    // 1. Initialize Library Handles
    CHECK_CUDNN(cudnnCreate(&cudnn));
    CHECK_CUBLAS(cublasCreate(&cublas));

    // 2. Create Descriptors (Opaque Handles)
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&conv1_out_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&pool1_out_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&conv2_out_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&pool2_out_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&conv1_bias_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&conv2_bias_desc));

    CHECK_CUDNN(cudnnCreateFilterDescriptor(&conv1_filter_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&conv2_filter_desc));

    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv1_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv2_desc));
    CHECK_CUDNN(cudnnCreatePoolingDescriptor(&pool_desc));
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&relu_desc));

    // 3. Configure Activation (ReLU) and Pooling (2x2 MaxPool)
    CHECK_CUDNN(cudnnSetActivationDescriptor(relu_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));
    CHECK_CUDNN(cudnnSetPooling2dDescriptor(pool_desc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 0, 0, 2, 2));

    // 4. Configure Filter Descriptors (Weight Shapes)
    // Format: [Out_Channels, In_Channels, H, W]
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(conv1_filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 6, 1, 5, 5));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(conv2_filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 16, 6, 5, 5));

    // 5. Configure Bias Descriptors (1 channel per filter)
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(conv1_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 6, 1, 1));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(conv2_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 16, 1, 1));

    // 6. Configure Convolution Math (Padding=2 for Conv1 to keep 28x28 size)
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv1_desc, 2, 2, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv2_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // 7. Initialize Tensor Shapes for Max Batch Size
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, max_batch_size, 1, 28, 28));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(conv1_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, max_batch_size, 6, 28, 28));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(pool1_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, max_batch_size, 6, 14, 14));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(conv2_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, max_batch_size, 16, 10, 10));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(pool2_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, max_batch_size, 16, 5, 5))  ;

    // 8. Allocate Intermediate Buffers (The Workspace)
    CHECK_CUDA(cudaMalloc(&d_conv1_out, max_batch_size * 6 * 28 * 28 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pool1_out, max_batch_size * 6 * 14 * 14 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_conv2_out, max_batch_size * 16 * 10 * 10 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_pool2_out, max_batch_size * 16 * 5 * 5 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fc1_out, max_batch_size * 120 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fc2_out, max_batch_size * 84 * sizeof(float)));

    // 9. Choose Conv Algorithm & Allocate cuDNN Workspace
    cudnnConvolutionFwdAlgoPerf_t perfResults;
    int returnedAlgoCount;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
                                           input_desc,
                                           conv1_filter_desc,
                                           conv1_desc,
                                           conv1_out_desc,
                                           1, // Request top 1 algorithm
                                           &returnedAlgoCount,
                                           &perfResults));
    conv1_algo = perfResults.algo;

    // Get Algo for Conv2 
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn, pool1_out_desc, conv2_filter_desc,
                                                       conv2_desc, conv2_out_desc, 1, &returnedAlgoCount, &perfResults));
    conv2_algo = perfResults.algo;  

    // 3. Calculate workspace for both and take the MAX
    size_t ws1, ws2;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_desc, conv1_filter_desc,
                                                        conv1_desc, conv1_out_desc, conv1_algo, &ws1));
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, pool1_out_desc, conv2_filter_desc,
                                                        conv2_desc, conv2_out_desc, conv2_algo, &ws2));

    workspace_size = std::max(ws1, ws2);
    CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
}

// --- LeNet5::forward implementation ---
void LeNet5::forward(float *d_input, int batch_size, float *d_output)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    
    /* code used for debugging
    float *h_input;
    CHECK_CUDA(cudaHostAlloc(&h_input, batch_size * 28 * 28 * sizeof(float), cudaHostAllocDefault));
    CHECK_CUDA(cudaMemcpy(h_input, d_input, batch_size * 28 * 28 * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < batch_size; i++)
    {
        float *imgI = h_input + i * 28 * 28;
        std::cout << "Image No. " << i << " values: " << std::endl;
        for (int j = 0; j < 28; j++)
        {
            for (int k = 0; k < 28; k++)
            {
                std::cout << imgI[j * 28 + k] << ",";
            }
            std::cout << std::endl;
        }
    }
    cudaFreeHost(h_input);
    */

    // 1. UPDATE TENSOR DESCRIPTORS FOR CURRENT BATCH SIZE
    // We assume descriptors were created in a setup() method; now we set the N (batch) dimension.
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 1, 28, 28));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(conv1_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 6, 28, 28));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(pool1_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 6, 14, 14));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(conv2_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 16, 10, 10));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(pool2_out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 16, 5, 5));

    // --- LAYER 1: CONV1 (5x5, 1->6) -> ReLU -> MaxPool (2x2) ---
    // Convolution
    CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, input_desc, d_input, conv1_filter_desc, d_conv1_w,
                            conv1_desc, conv1_algo, workspace, workspace_size, &beta, conv1_out_desc, d_conv1_out));
    // Add Bias
    CHECK_CUDNN(cudnnAddTensor(cudnn, &alpha, conv1_bias_desc, d_conv1_b, &alpha, conv1_out_desc, d_conv1_out));
    // Activation (ReLU)
    CHECK_CUDNN(cudnnActivationForward(cudnn, relu_desc, &alpha, conv1_out_desc, d_conv1_out, &beta, conv1_out_desc, d_conv1_out));
    // Pooling
    CHECK_CUDNN(cudnnPoolingForward(cudnn, pool_desc, &alpha, conv1_out_desc, d_conv1_out, &beta, pool1_out_desc, d_pool1_out));

    // --- LAYER 2: CONV2 (5x5, 6->16) -> ReLU -> MaxPool (2x2) ---
    // Convolution
    CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, pool1_out_desc, d_pool1_out, conv2_filter_desc, d_conv2_w,
                            conv2_desc, conv2_algo, workspace, workspace_size, &beta, conv2_out_desc, d_conv2_out));
    // Add Bias
    CHECK_CUDNN(cudnnAddTensor(cudnn, &alpha, conv2_bias_desc, d_conv2_b, &alpha, conv2_out_desc, d_conv2_out));
    // Activation (ReLU)
    CHECK_CUDNN(cudnnActivationForward(cudnn, relu_desc, &alpha, conv2_out_desc, d_conv2_out, &beta, conv2_out_desc, d_conv2_out));
    // Pooling
    CHECK_CUDNN(cudnnPoolingForward(cudnn, pool_desc, &alpha, conv2_out_desc, d_conv2_out, &beta, pool2_out_desc, d_pool2_out)) ;

    // --- LAYER 3: FC1 (Fully Connected 400 -> 120) ---
    // Matrix Multiplication: [120 x 400] * [400 x batch] -> [120 x batch]
    // We use CUBLAS_OP_T for weights because they are Row-Major in PyTorch's binary export.
    CHECK_CUBLAS(cublasSgemm(cublas,
                CUBLAS_OP_T, // Transpose Row-Major [120x400] to Column-Major
                CUBLAS_OP_N, // Data is [400xBatch]
                120, batch_size, 400,
                &alpha,
                d_fc1_w, 400, // LDA is the "width" of the original row-major matrix
                d_pool2_out, 400,
                &beta,
                d_fc1_out, 120));
  
    // Apply Bias and ReLU via custom CUDA kernel
    dim3 block_fc1(threads__per_block);
    dim3 grid_fc1((120 + threads__per_block - 1) / threads__per_block, batch_size);
    apply_bias_relu_kernel(grid_fc1, block_fc1, d_fc1_out, d_fc1_b, 120, batch_size);

    // --- LAYER 4: FC2 (Fully Connected 120 -> 84) ---
    // Matrix Multiplication: [84 x 120] * [120 x batch] -> [84 x batch]
    CHECK_CUBLAS(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                84, batch_size, 120,
                &alpha, 
                d_fc2_w, 120,
                d_fc1_out, 120,
                &beta, d_fc2_out, 84));
    // Apply Bias and ReLU
    dim3 block_fc2(threads__per_block);
    dim3 grid_fc2((84 + threads__per_block - 1) / threads__per_block, batch_size);
    apply_bias_relu_kernel(grid_fc2, block_fc2, d_fc2_out, d_fc2_b, 84, batch_size);

    // --- LAYER 5: FC3 (Fully Connected 84 -> 62) - FINAL OUTPUT ---
    // Matrix Multiplication: [62 x 84] * [84 x batch] -> [62 x batch]
    CHECK_CUBLAS(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                62, batch_size, 84,
                &alpha, d_fc3_w, 84,
                d_fc2_out, 84,
                &beta, d_output, 62));
    // Apply Final Bias Only (No ReLU)
    dim3 block_fc3(threads__per_block);
    dim3 grid_fc3((62 + threads__per_block - 1) / threads__per_block, batch_size);
    apply_bias_only_kernel(grid_fc3, block_fc3, d_output, d_fc3_b, 62, batch_size);

    cudaDeviceSynchronize(); // make sure GPU is complete before CPU intepret the results
}

float *LeNet5::loadFileToDevice(const std::string &modelPath, const std::string &filename, int numElements)
{
    size_t bytes = numElements * sizeof(float);
    float *h_ptr;

    CHECK_CUDA(cudaHostAlloc(&h_ptr, bytes, cudaHostAllocDefault));

    fs::path fullPath = fs::path(modelPath) / filename;
    

    std::ifstream file(fullPath, std::ios::binary);
    if (!file)
    {
        std::cerr << "Fatal Error: Missing " << fullPath << std::endl;
        exit(EXIT_FAILURE);
    }
    file.read(reinterpret_cast<char *>(h_ptr), bytes);
    file.close();

    float *d_ptr;
    CHECK_CUDA(cudaMalloc(&d_ptr, bytes));
    CHECK_CUDA(cudaMemcpy(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice));

    // print the first three floats to make sure match with training size result
    std::cout << "weights " << filename << ", first three floats: " << h_ptr[0] << ", " << h_ptr[1] << ", " << h_ptr[2] << std::endl;

    cudaFreeHost(h_ptr);
    return d_ptr;
}

void LeNet5::loadModel(const std::string &modelPath)
{
    std::cout << "Loading LeNet-5 weights from " << modelPath <<  ",  ..." << std::endl;

    d_conv1_w = loadFileToDevice(modelPath, "conv1_w.bin", CONV1_W_SIZE);
    d_conv1_b = loadFileToDevice(modelPath, "conv1_b.bin", CONV1_B_SIZE);

    d_conv2_w = loadFileToDevice(modelPath, "conv2_w.bin", CONV2_W_SIZE);
    d_conv2_b = loadFileToDevice(modelPath, "conv2_b.bin", CONV2_B_SIZE);

    d_fc1_w = loadFileToDevice(modelPath, "fc1_w.bin", FC1_W_SIZE);
    d_fc1_b = loadFileToDevice(modelPath, "fc1_b.bin", FC1_B_SIZE);

    d_fc2_w = loadFileToDevice(modelPath, "fc2_w.bin", FC2_W_SIZE);
    d_fc2_b = loadFileToDevice(modelPath, "fc2_b.bin", FC2_B_SIZE);

    d_fc3_w = loadFileToDevice(modelPath, "fc3_w.bin", FC3_W_SIZE);
    d_fc3_b = loadFileToDevice(modelPath, "fc3_b.bin", FC3_B_SIZE);

    std::cout << "LeNet5 model loaded" << std::endl;
}

void LeNet5::cleanup()
{
    // 1. Free Device Data Buffers (Intermediate Layers)
    cudaFree(d_conv1_out);
    cudaFree(d_pool1_out);
    cudaFree(d_conv2_out);
    cudaFree(d_pool2_out);
    cudaFree(d_fc1_out);
    cudaFree(d_fc2_out);

    // 2. Free Weight/Bias Buffers (Loaded from .bin)
    cudaFree(d_conv1_w);
    cudaFree(d_conv1_b);
    cudaFree(d_conv2_w);
    cudaFree(d_conv2_b);
    cudaFree(d_fc1_w);
    cudaFree(d_fc1_b);
    cudaFree(d_fc2_w);
    cudaFree(d_fc2_b);
    cudaFree(d_fc3_w);
    cudaFree(d_fc3_b);

    // 3. Free cuDNN Workspace
    cudaFree(workspace);

    // 4. Destroy cuDNN Descriptors
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(conv1_out_desc);
    cudnnDestroyTensorDescriptor(pool1_out_desc);
    cudnnDestroyTensorDescriptor(conv2_out_desc);
    cudnnDestroyTensorDescriptor(pool2_out_desc);
    cudnnDestroyTensorDescriptor(conv1_bias_desc);
    cudnnDestroyTensorDescriptor(conv2_bias_desc);

    cudnnDestroyFilterDescriptor(conv1_filter_desc);
    cudnnDestroyFilterDescriptor(conv2_filter_desc);

    cudnnDestroyConvolutionDescriptor(conv1_desc);
    cudnnDestroyConvolutionDescriptor(conv2_desc);
    cudnnDestroyPoolingDescriptor(pool_desc);
    cudnnDestroyActivationDescriptor(relu_desc);

    // 5. Destroy Library Handles
    if (cudnn)
        cudnnDestroy(cudnn);
    if (cublas)
        cublasDestroy(cublas);

    std::cout << "LeNet-5 Resources Cleaned Up." << std::endl;
}
