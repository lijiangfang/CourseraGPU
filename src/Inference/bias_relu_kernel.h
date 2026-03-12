
#include <cuda_runtime.h>

__host__ void apply_bias_relu_kernel(dim3 gridDim, dim3 blockDim, float* data, const float* bias, int num_features, int batch_size) ;
__host__ void apply_bias_only_kernel(dim3 gridDim, dim3 blockDim, float* data, const float* bias, int num_features, int batch_size) ;