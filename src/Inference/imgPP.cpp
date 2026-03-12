
#include <cuda_runtime.h>
#include <npp.h>
#include <nppi_geometry_transforms.h>  
#include <nppi_color_conversion.h>     
#include <nppi_data_exchange_and_initialization.h>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <algorithm>
#include "cucheck.h"

// Include stb_image for file I/O
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


// Structure to hold image data on the CPU
struct HostImage
{
    unsigned char *pData; // 8-bit RGB pixels from file
    std::string filename;
    int width, height, channels;
};

/**
 * NPP Pipeline: Upload -> RGB to Gray -> Batch Resize -> Normalize -> pack into 1D float* for LeNet-5
 */
void normalizeImages(std::vector<HostImage> &batch, float *d_inputBuffer, int targetW, int targetH)
{
    int batchSize = batch.size();
    if (batchSize == 0)
        return;

    // Vectors to track device pointers and steps (pitches)
    std::vector<Npp8u *> d_src(batchSize), d_gray(batchSize), d_resized(batchSize);
    std::vector<Npp32f *> d_norm(batchSize);
    std::vector<int> srcStep(batchSize), grayStep(batchSize), resStep(batchSize), normStep(batchSize);

    // ALLOCATION & UPLOAD
    for (int i = 0; i < batchSize; ++i)
    {
        // Source RGB (3-channel)
        d_src[i] = nppiMalloc_8u_C3(batch[i].width, batch[i].height, &srcStep[i]);
        CHECK_CUDA(cudaMemcpy2D(d_src[i], srcStep[i], batch[i].pData, batch[i].width * 3,
                     batch[i].width * 3, batch[i].height, cudaMemcpyHostToDevice));

        // Intermediate Gray & Resized (1-channel)
        d_gray[i] = nppiMalloc_8u_C1(batch[i].width, batch[i].height, &grayStep[i]);
        d_resized[i] = nppiMalloc_8u_C1(targetW, targetH, &resStep[i]);

        // Final Normalized (1-channel Float)
        d_norm[i] = nppiMalloc_32f_C1(targetW, targetH, &normStep[i]);
    }

    // STAGE 1: RGB TO GRAY (Point-wise)
    for (int i = 0; i < batchSize; ++i)
    {
        NppiSize size = {batch[i].width, batch[i].height};
        nppiRGBToGray_8u_C3C1R(d_src[i], srcStep[i], d_gray[i], grayStep[i], size);
    }

    // STAGE 2: BATCH RESIZE
    // Use Advanced descriptors for variable source sizes
    NppiSize dstSize = {targetW, targetH};
    NppiImageDescriptor *d_srcBatch = nullptr, *d_dstBatch = nullptr;
    NppiResizeBatchROI_Advanced *d_roiBatch = nullptr;
    if (batchSize > 1)
    {
        std::vector<NppiImageDescriptor> h_srcBatch(batchSize);
        std::vector<NppiImageDescriptor> h_dstBatch(batchSize);
        std::vector<NppiResizeBatchROI_Advanced> h_roiBatch(batchSize);

        for (int i = 0; i < batchSize; ++i)
        {
            h_srcBatch[i].pData = d_gray[i];
            h_srcBatch[i].nStep = grayStep[i];
            h_srcBatch[i].oSize = {batch[i].width, batch[i].height};

            h_dstBatch[i].pData = d_resized[i];
            h_dstBatch[i].nStep = resStep[i];
            h_dstBatch[i].oSize = {targetW, targetH};

            // Per-image ROI mapping
            h_roiBatch[i].oSrcRectROI = {0, 0, batch[i].width, batch[i].height};
            h_roiBatch[i].oDstRectROI = {0, 0, targetW, targetH};
        }

        // Allocate and copy these 3 arrays to Device
        cudaMalloc(&d_srcBatch, batchSize * sizeof(NppiImageDescriptor));
        cudaMalloc(&d_dstBatch, batchSize * sizeof(NppiImageDescriptor));
        cudaMalloc(&d_roiBatch, batchSize * sizeof(NppiResizeBatchROI_Advanced));

        cudaMemcpy(d_srcBatch, h_srcBatch.data(), batchSize * sizeof(NppiImageDescriptor), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dstBatch, h_dstBatch.data(), batchSize * sizeof(NppiImageDescriptor), cudaMemcpyHostToDevice);
        cudaMemcpy(d_roiBatch, h_roiBatch.data(), batchSize * sizeof(NppiResizeBatchROI_Advanced), cudaMemcpyHostToDevice);

        nppiResizeBatch_8u_C1R_Advanced(targetW, targetH, d_srcBatch, d_dstBatch, d_roiBatch, batchSize, NPPI_INTER_LINEAR);

        // Call the Advanced version
    }
    else
    {
        NppiSize srcSize = {batch[0].width, batch[0].height};
        NppiRect srcROI = {0, 0, batch[0].width, batch[0].height};
        NppiRect dstROI = {0, 0, targetW, targetH};
        nppiResize_8u_C1R(d_gray[0], grayStep[0], srcSize, srcROI,
                          d_resized[0], resStep[0], dstSize, dstROI,
                          NPPI_INTER_LINEAR);
    }

    // STAGE 3: CONVERT & NORMALIZE
    for (int i = 0; i < batchSize; ++i)
    {
        // 1. Convert 8-bit unsigned to 32-bit float
        // d_resized[i] is 0-255 (8u), d_norm[i] becomes 0.0f-255.0f (32f)
        nppiConvert_8u32f_C1R(d_resized[i], resStep[i], d_norm[i], normStep[i], dstSize);

        // 2. Perform the inversion: 255.0f - pixel
        // Note: nppiSubC_32f_C1R performs (dst = src - constant).
        // To get (255 - pixel), we use nppiSubC_32f_C1R followed by negation,
        // or simply use MulC and AddC:
        // Better way for (255.0f - pixel):
        // pixel = -1.0 * pixel + 255.0
        nppiMulC_32f_C1IR(-1.0f, d_norm[i], normStep[i], dstSize);
        nppiAddC_32f_C1IR(255.0f, d_norm[i], normStep[i], dstSize);

        // 2. Apply Normalization Math: (scaled_pixel - 0.1736) / 0.3317
        // Factor = (1/255) / 0.3317 = 0.011822
        // Offset = -(0.1736 / 0.3317) = -0.5233
        nppiMulC_32f_C1IR(0.011822f, d_norm[i], normStep[i], dstSize);
        nppiAddC_32f_C1IR(-0.5233f, d_norm[i], normStep[i], dstSize);

        // 3. Copy directly to slot (No Transpose needed anymore!)
        float *slot = d_inputBuffer + (i * 28 * 28);
        nppiCopy_32f_C1R(d_norm[i], normStep[i], slot, 28 * sizeof(float), {28, 28});
    }

    cudaDeviceSynchronize();
    for (int i = 0; i < batchSize; ++i)
    {
        // code used for debugging delete the following block when debug is done
        // float *slot = d_inputBuffer + (i * 28 * 28);
        // float host_check[28 * 28];
        // cudaMemcpy(host_check, slot, 28 * 28 * sizeof(float), cudaMemcpyDeviceToHost);
        // std::cout << "First pixel in slot: " << host_check[0] << std::endl;

        // Free device pointers
        nppiFree(d_src[i]);
        nppiFree(d_gray[i]);
        nppiFree(d_resized[i]);
        nppiFree(d_norm[i]);
    }
    // Also free the batch descriptors
    cudaFree(d_srcBatch);
    cudaFree(d_dstBatch);
    cudaFree(d_roiBatch);
}

std::tuple<std::vector<std::string>, int> getBatchInputs(std::vector<std::string> filenames, float *d_inputBuffer)
{
    std::vector<HostImage> batch;
    std::vector<std::string> validFileNames;
    const int targetW = 28;
    const int targetH = 28;

    for (const auto &f : filenames)
    {
        HostImage img;
        img.pData = stbi_load(f.c_str(), &img.width, &img.height, &img.channels, 3);
        if (img.pData)
        {
            batch.push_back(img);
            validFileNames.push_back(f);
        }
        else
        {
            std::cerr << "Failed to load: " << f << std::endl;
        }
    }

    if (!batch.empty())
    {
        normalizeImages(batch, d_inputBuffer, targetW, targetH);

        /* code used for debugging 
        float *h_input;
        cudaDeviceSynchronize();
        cudaHostAlloc(&h_input, batch.size() * 28 * 28 * sizeof(float), cudaHostAllocDefault);
        cudaMemcpy(h_input, d_inputBuffer, batch.size() * 28 * 28 * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < batch.size(); i++)
        {
            float *imgI = h_input + i * 28 * 28;
            std::cout << "Image: " << filenames[i]  << " values: " << std::endl;
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
    }

    for (int i = 0; i < batch.size(); ++i)
    {
      stbi_image_free(batch[i].pData);
    }

    return {validFileNames, batch.size()};
}


