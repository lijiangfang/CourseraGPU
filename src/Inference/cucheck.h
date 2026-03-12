// Macro for CUDA and cuBLAS
#ifndef CHECK_CUDA
#define CHECK_CUDA(call)    \
    {   \
        cudaError_t err = (call);   \
        if (err != cudaSuccess) \
        {   \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " in " << __FILE__ << " : " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        }   \
    }
#endif

// Macro for cuDNN
#ifndef CHECK_CUDNN
#define CHECK_CUDNN(call)   \
    {   \
        cudnnStatus_t status = (call);      \
        if (status != CUDNN_STATUS_SUCCESS) \
        {   \
            std::cerr << "CUDNN Error: " << cudnnGetErrorString(status) << " in " << __FILE__ << " : " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        }   \
    }
#endif

// Marco for cuBLAS
#ifndef CHECK_CUBLAS
#include <cublas_v2.h>

inline const char *cublasGetErrorString(cublasStatus_t status)
{
    switch (status)
    {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    default:
        return "UNKNOWN_CUBLAS_ERROR";
    }
}

#define CHECK_CUBLAS(call) \
    {                      \
        cublasStatus_t status = (call); \
        if (status != CUBLAS_STATUS_SUCCESS)    \
        {   \
            std::cerr << "CUBLAS Error: " << cublasGetErrorString(status) << " in " << __FILE__ " : " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        }   \
    }
#endif 