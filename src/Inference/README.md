This directory contains the C++ code used to run evaluation on the trained model.

### Files

- `lenet5.h`: Header file for the `LeNet5` class.  
- `lenet5.cpp`: C++ implementation of the `LeNet5` class. The code loads the model weights in the `setup` method. The `forward` method performs batch inference. It uses **cuDNN** and **cuBLAS**, as well as custom CUDA kernels.  
- `bias_relu_kernel.h`: Header file for CUDA kernels.  
- `bias_relu_kernel.cu`: CUDA kernel implementations.  
- `imgPP.h`: Header file for image pre-processing before passing images to `LeNet5` for evaluation.  
- `imgPP.cpp`: Reads `.jpeg` / `.jpg` files and preprocesses the images. It uses the **CUDA NPP** library.  
- `ocr.cpp`: Handles command-line options, calls image preprocessing, then calls `LeNet5::forward`. Processes the output from `LeNet5` and prints the results to standard output.  
- `cucheck.h`: Macros for checking CUDA, cuDNN, and cuBLAS error codes.

---

### Build

Type the following command in a terminal:

```bash
make
```

### run

Type the following command in a terminal:

```bash
./ocr.exe [-relaxed] [-inputD <input_directory>] [-modelD <model_directory>] [-batchSize <batch_size>] [-files file1.jpg,file2.jpg,...]
```

#### Defaults

- **Input directory:** `../../inputs/`  
- **Model directory:** `../../model/`  
- **Files:** if not specified, all `.jpeg` / `.jpg` files in the input directory are used  
- **Batch size:** 5  
- **-relaxed:** when selected, prediction correctness is evaluated more leniently (e.g., `O`, `o`, and `0` are considered equivalent)