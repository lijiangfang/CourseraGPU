#include <iostream>
#include <fstream>
#include <string>
#include "bias_relu_kernel.h"
#include <cudnn.h>
#include <cublas_v2.h>

class LeNet5
{
public:
    // --- Model Architecture Constants ---
    // Conv1: 6 filters, 1 input channel, 5x5 kernel
    static const int CONV1_W_SIZE = 6 * 1 * 5 * 5;
    static const int CONV1_B_SIZE = 6;

    // Conv2: 16 filters, 6 input channels, 5x5 kernel
    static const int CONV2_W_SIZE = 16 * 6 * 5 * 5;
    static const int CONV2_B_SIZE = 16;

    // FC1: 16*5*5 (400) inputs -> 120 outputs
    static const int FC1_W_SIZE = 120 * 16 * 5 * 5;
    static const int FC1_B_SIZE = 120;

    // FC2: 120 inputs -> 84 outputs
    static const int FC2_W_SIZE = 84 * 120;
    static const int FC2_B_SIZE = 84;

    // FC3: 84 inputs -> 62 outputs (0-9, A-Z, a-z)
    static const int FC3_W_SIZE = 62 * 84;
    static const int FC3_B_SIZE = 62;

    LeNet5();
    ~LeNet5();

    void setup(std::string &modelPath, int max_batch_size);
    void forward(float *d_input, int batch_size, float *d_output);
    
private:
    const int threads__per_block = 256; // For custom kernels, we will use 256 threads per block

    // --- model weights Pointers ---
    float *d_conv1_w, *d_conv1_b;
    float *d_conv2_w, *d_conv2_b;
    float *d_fc1_w, *d_fc1_b;
    float *d_fc2_w, *d_fc2_b;
    float *d_fc3_w, *d_fc3_b;

    // Library Handles
    cudnnHandle_t  cudnn;
    cublasHandle_t cublas;

    // Tensor/Filter Descriptors (Metadata)
    cudnnTensorDescriptor_t input_desc, conv1_out_desc, pool1_out_desc;
    cudnnTensorDescriptor_t conv2_out_desc, pool2_out_desc;
    cudnnFilterDescriptor_t conv1_filter_desc, conv2_filter_desc;
    cudnnTensorDescriptor_t conv1_bias_desc, conv2_bias_desc;
    cudnnConvolutionDescriptor_t conv1_desc, conv2_desc;
    cudnnPoolingDescriptor_t     pool_desc;
    cudnnActivationDescriptor_t  relu_desc;

    // Performance Buffers
    void*  workspace;
    size_t workspace_size;
    cudnnConvolutionFwdAlgo_t conv1_algo, conv2_algo;

    // Intermediate Data Buffers (The actual floats between layers)
    float *d_conv1_out, *d_pool1_out, *d_conv2_out, *d_pool2_out;
    float *d_fc1_out, *d_fc2_out;

    float *loadFileToDevice(const std::string &modelPath, const std::string &filename, int numElements);
    void cleanup();
    void loadModel(const std::string &modelPath);
};
