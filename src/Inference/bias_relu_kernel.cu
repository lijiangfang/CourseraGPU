
#include "bias_relu_kernel.h"

__global__ void bias_relu_kernel(float* data, const float* bias, int num_features) {
    int feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y; // Use Y for batch dimension

    if (feat_idx < num_features) {
        int index = batch_idx * num_features + feat_idx;
        float val = data[index] + bias[feat_idx];
        data[index] = (val > 0.0f) ? val : 0.0f; // Manual ReLU
    }
}

__host__ void apply_bias_relu_kernel(dim3 gridDim, dim3 blockDim, float* data, const float* bias, int num_features, int batch_size) 
{
    bias_relu_kernel<<<gridDim, blockDim>>>(data, bias, num_features);
}

__global__ void bias_only_kernel(float* data, const float* bias, int num_features, int batch_size) {
    int feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y;

    if (feat_idx < num_features && batch_idx < batch_size) {
        int index = batch_idx * num_features + feat_idx;
        data[index] += bias[feat_idx]; // No ReLU for the final output layer
    }
}

__host__ void apply_bias_only_kernel(dim3 gridDim, dim3 blockDim, float* data, const float* bias, int num_features, int batch_size) 
{
    bias_only_kernel<<<gridDim, blockDim>>>(data, bias, num_features, batch_size);
}