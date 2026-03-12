This directory contains the C++ code to run evaluation on the model we trained.

# Files
*lenet5.h: the header file for LeNet5 class
*lenet5.cpp: the C++ implementation for LetNet5 class. The code loads the model weights in setup method. The forward method does the batch inferencing. It builds on cuDNN and cuBLAS as well as own CUDA kernel.
*bias_relu_kernel.h: header file for CUDA kernels.
*bias_relu_kernel.cu: CUDA kernel implementations.
*imgPP.h: header file for image pre-processing before the images are passed to LeNet5 for evaluation.
*imgPP.cpp: reads .jpeg/.jpg files and preprecesses the images. It builds on CUDA NPPI library.
*ocr.cpp: handles the command line options, calls image preprocessing, and then calls LeNet5 forward. After that, processes the output from LeNet5 and print the result to standard output.
*cucheck.h: some macroes for CUDA, cuDNN, and cuBLAS error code check.

# build
type make in a command line

# run
type following in command line:
/ocr.exe [-relaxed] [-inputD <input_directory>] [-modelD <model_directory>] [-batchSize <batch_size>] [-files file1.jpg,file2.jpg,...]

Example: ./ocr -inputD ./input -modelD ./model -max_batch 5

Example with specific files: ./ocr -inputD ./input -modelD ./model -max_batch 5 -files img1.jpg,img2.jpg

default input directory ../../inputs/, default model path: ../../model/. no specifying files means all .jpeg/.jpg files under inputs directory.

-relaxed: when this option is selected, the prediction correctness is evalued more relaxed. e.g. O, o, and 0 are considered the same

default batch size is 5.


