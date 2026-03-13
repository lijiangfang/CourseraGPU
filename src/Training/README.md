This directory contains the Python code used to train the model. The training code should work on both GPU and CPU. However, in the Coursera lab environment for this class, PyTorch is not installed. Therefore, the training was performed on the CPU of my MacBook.

Modify the following variables as needed:

- `data_path`: where the training data is stored
- `model_path`: where the model parameters will be written after training
- `TOTAL_EPOCHS`: the number of epochs to train

To run the training loop, type the following command in the command line:

```bash
python3 LeNet5.py
