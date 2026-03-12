import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

data_path = '../../data'  
model_path = '../../model/'
TOTAL_EPOCHS = 5

# 1. Define LeNet-5 Architecture 
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # Layer 1: Conv 5x5, 1 input channel (gray), 6 output channels
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2) 
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 2: Conv 5x5, 6 input, 16 output
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 3: Fully Connected
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 62) # 62 Classes (0-9, A-Z, a-z)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

# 2. Setup Data (EMNIST ByClass)
transform = transforms.Compose([
    transforms.ToTensor(),
    # EMNIST is transposed by default; this flip/rotate fix is helpful for C++ alignment
    lambda x: x.transpose(1, 2),
    # Standardize using EMNIST ByClass mean/std (calculated from the dataset)
    transforms.Normalize((0.1736,), (0.3317,)) 
])

train_set = torchvision.datasets.EMNIST(root=data_path, split='byclass', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

# 3. Training Loop (Run for 5-10 epochs)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Training on {device}...")
model.train()
for epoch in range(TOTAL_EPOCHS): # 5 epochs is enough for a baseline
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if i % 1000 == 0:
            print(f"Epoch [{epoch+1}/{TOTAL_EPOCHS}], Step [{i}], Loss: {loss.item():.4f}")

# 4. Export to Binary for C++ / CUDA
print("Exporting weights to .bin files...")
model.eval()

def save_bin(tensor, name):
    # Ensure float32 for C++ compatibility
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    arr.tofile(model_path + f"{name}.bin")
    print(f"Saved {name}.bin | Shape: {arr.shape}, first three floats: {arr.flatten()[:3]}")

save_bin(model.conv1.weight, "conv1_w")
save_bin(model.conv1.bias, "conv1_b")
save_bin(model.conv2.weight, "conv2_w")
save_bin(model.conv2.bias, "conv2_b")
save_bin(model.fc1.weight, "fc1_w")
save_bin(model.fc1.bias, "fc1_b")
save_bin(model.fc2.weight, "fc2_w")
save_bin(model.fc2.bias, "fc2_b")
save_bin(model.fc3.weight, "fc3_w")
save_bin(model.fc3.bias, "fc3_b")

print("All weights exported. ")
