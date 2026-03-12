
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>
#include <tuple>
#include <algorithm>
#include "imgPP.h"
#include "lenet5.h"

namespace fs = std::filesystem;

struct AppConfig
{
    std::string modelPath = "../../model/";
    std::string inputPath = "../../inputs/";
    std::vector<std::string> fileList;
    int maxBatchSize = 8;
    bool relaxed = false;
};

void printUsage()
{
    std::cout << "Usage: ./ocr -relaxed -inputD <input_directory> -modelD <model_directory> -batchSize <batch_size> [-files file1.jpg,file2.jpg,...]" << std::endl;
    std::cout << "Example: ./ocr -inputD ./input -modelD ./model -max_batch 5" << std::endl;
    std::cout << "Example with specific files: ./ocr -inputD ./input -modelD ./model -max_batch 5 -files img1.jpg,img2.jpg" << std::endl;
    std::cout << "default input directory ../../inputs/, default model path: ../../model/. no specifying files means all .jpeg/.jpg files under inputs directory" << std::endl;
    std::cout << "-relaxed: when this option is selected, the prediction correctness is evalued more relaxed. e.g. O, o, and 0 are considered the same" << std::endl;
}

AppConfig parseArgs(int argc, char *argv[])
{
    AppConfig config;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h")
        {
            printUsage();
            exit(0);
        }
        else if (arg == "-relaxed")
        {
            config.relaxed = true;
        }
        else if (arg == "-inputD" && i + 1 < argc)
        {
                config.inputPath = argv[++i];
        }
        else if (arg == "-modelD" && i + 1 < argc)
        {
            config.modelPath = argv[++i];
        }
        else if (arg == "-batchSize" && i + 1 < argc)
        {
            config.maxBatchSize = std::stoi(argv[++i]);
        }
        else if (arg == "-files" && i + 1 < argc)
        {
            std::string filesStr = argv[++i];
            std::stringstream ss(filesStr);
            std::string item;
            // Split by comma
            while (std::getline(ss, item, ','))
            {
                if (!item.empty())
                    config.fileList.push_back(item);
            }
        }
    }
    return config;
}

std::vector<std::string> getJpegFiles(const std::string &directoryPath)
{
    std::vector<std::string> jpegFiles;

    try
    {
        if (fs::exists(directoryPath) && fs::is_directory(directoryPath))
        {
            for (const auto &entry : fs::directory_iterator(directoryPath))
            {
                // Check if it's a regular file
                if (fs::is_regular_file(entry))
                {
                    std::string ext = entry.path().extension().string();

                    // Convert extension to lowercase for robust matching (.JPEG vs .jpeg)
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                    if (ext == ".jpeg" || ext == ".jpg")
                    {
                        // Use .filename() to get just "image.jpg"
                        // or .string() for the full path "./input/image.jpg"
                        jpegFiles.push_back(entry.path().filename().string());
                    }
                }
            }
        }
    }
    catch (const fs::filesystem_error &e)
    {
        std::cerr << "Error accessing directory: " << e.what() << std::endl;
    }

    return jpegFiles;
}

float computeConfidence (float *logits, int idxMax)
{
    float max_val = logits[idxMax];

    // Compute exponentials and their sum
    float sum = 0.0f;
    for (int i = 0; i < 62; ++i)
    {
        sum += std::exp(logits[i] - max_val);
    }

    float confidence = 1.0 / sum;

    return confidence;
}  

int processModelOutput(float *h_output,float *d_output, int batch_size, std::vector<std::string> &batchFileNames, bool relaxed)
{
    // This function can be used to copy the output back to host and interpret results
    // For example, you might want to find the index of the max value for each item in the batch
    // and map that index to a character (0-9, A-Z, a-z)
    const char* labels =        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    const char *labels_alt1 =   "OlZ34S6789ABcDEFGH1JkLmnopQRsTuvwxYzabCdefgh1jKlMNOPqrStUVWXyZ";
    const char *labels_alt2 =   "oiz34s6789ABCDEFGHlJKLMN0PQR5TUVWXY2abcdefghljklmn0pqr5tuvwxy2";
    int correctCount = 0;

    cudaMemcpy(h_output, d_output, batch_size * 62 * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < batch_size; ++i)
    {
        float *logits = h_output + (i * 62);
        /* code used for debugging
        for (int j = 0; j < 62; j++)
        {
            std::cout << "class: " << j << ", logits: " <<  logits[j] << std::endl;
        }
        */

        std::vector<std::pair<float, int>> paired_logits;
        for (int j = 0; j < 62; ++j)
        {
            paired_logits.push_back({logits[j], j});
        }

        // 2. Partial sort the top 3 based on the value (first element of pair)
        std::partial_sort(paired_logits.begin(), paired_logits.begin() + 3, paired_logits.end(),
                          [](const auto &a, const auto &b) { return a.first > b.first; });
        // 3. Display results
        for (int j = 0; j < 3; ++j)
        {
            int idx = paired_logits[j].second;
            std::cout
                << "Top " << j + 1 << ": Class " << idx << " -- map to " << labels[idx]
                << " with Logit " << paired_logits[j].first << "\n";
        }

        int predicted_index = std::distance(logits, std::max_element(logits, logits + 62));
        float confidence = computeConfidence(logits, predicted_index);
        char predicted_char = labels[predicted_index];

        // check if prediction is correct
        fs::path fullpath(batchFileNames[i]);
        std::string stem = fullpath.stem();
        bool isCorrect;
        if (!relaxed)
        {
            isCorrect = stem[0] == predicted_char;
        }
        else
        {
            // e.g. O, o, and 0 are very hard to tell the difference in hand writing without more context
            // similarly U and u etc
            isCorrect = (stem[0] == predicted_char) || (stem[0] == labels_alt1[predicted_index]) || (stem[0] == labels_alt2[predicted_index]);
        }

        std::string correctly = isCorrect ? "correctly" : "incorrectly";
        if (isCorrect)
        {
            correctCount++;
            std::cout << "\033[1;32m"; // bold Green
        }
        else
        {
            std::cout << "\033[1;31m"; // bold Red
        }
        std::cout << "Image: " << batchFileNames[i] << " -> " << correctly << " predicted: " << predicted_char << " , Confidence: " << confidence << "\033[0m"
            << std::endl
            << std::endl;
    }

    return correctCount;
}

int main(int argc, char *argv[])
{
    int totalCount = 0;
    int correctCount = 0;

    AppConfig config = parseArgs(argc, argv);

    std::vector<std::string> fileList;

    if (!config.fileList.empty())
    {
        fileList = config.fileList;
        }
        else
        {
            fileList = getJpegFiles(config.inputPath);
        }

        std::cout << "Input Directory:" << config.inputPath << std::endl;
        std::cout << "Model Path:" << config.modelPath << std::endl;
        std::cout << "Iput File Names:" << std::endl;
        int fileCount = 0;
        for (const auto &f : fileList)
        {
            fileCount++;
            std::cout << f << ", ";
            if (fileCount % 20 == 0)
            {
                std::cout << std::endl;
            } 
        }
        std::cout << std::endl;

        LeNet5 model;
        model.setup(config.modelPath, config.maxBatchSize);

        std::vector<fs::path> inputFiles;
        for (const auto &f : fileList)
        {
            fs::path fullPath = fs::path(config.inputPath) / f;
            inputFiles.push_back(fullPath);
        }

        // Load images into batch and run inference in batches of maxBatchSize
        float *d_input;
        cudaMalloc(&d_input, config.maxBatchSize * 28 * 28 * sizeof(float));
        float *d_output;
        cudaMalloc(&d_output, config.maxBatchSize * 62 * sizeof(float));             // Output buffer for max batch size
        float *h_output; // Host buffer for output
        cudaHostAlloc((void **)&h_output, config.maxBatchSize * 62 * sizeof(float), cudaHostAllocDefault);

        std::vector<std::string> batchFileNames;
        for (const auto &f : inputFiles)
        {
            batchFileNames.push_back(f.string());
            if (batchFileNames.size() == config.maxBatchSize)
            {
                auto [validFileNames, batch_size] = getBatchInputs(batchFileNames, d_input);
                if (batch_size > 0)
                {
                    model.forward(d_input, batch_size, d_output);
                    // Process the batch output as needed (e.g., copy back to host, interpret results)
                    correctCount += processModelOutput(h_output, d_output, batch_size, validFileNames, config.relaxed);
                    totalCount += validFileNames.size();
                }
                batchFileNames.clear();
            }
        }

        // Process any remaining files in the last batch
        if (!batchFileNames.empty())
        {
            auto [validFileNames, batch_size] = getBatchInputs(batchFileNames, d_input);
            if (batch_size > 0)
            {
                model.forward(d_input, batch_size, d_output);

                // Process the batch output as needed
                correctCount += processModelOutput(h_output, d_output, batch_size, validFileNames, config.relaxed);
                totalCount += validFileNames.size();
            }
        }
        cudaFree(d_output); // Free the output buffer after all batches are processed
        cudaFree(d_input);  // Free the input buffer (in case it wasn't freed in the loop)
        cudaFreeHost(h_output); // Free the host output buffer

        std::cout << "\033[1;32m"
                <<  "Total " << totalCount << " images, " << correctCount << " predictions correct" << std::endl;
        if (totalCount > 0)
        {
            std::cout << "Accuracy rate: " << ((float)correctCount) / ((float)totalCount) << std::endl;
        }
        std::cout << "\033[0m";
        std::cout << "Processing complete. " << std::endl;
        return 0;
}