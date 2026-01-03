import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import torch.onnx


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

num_classes = 184  # Leafsnap Field dataset

# Create ResNet18 architecture
model = models.resnet18(weights=None)  # weights=None because we load our own
model.fc = nn.Linear(512, num_classes)

model = model.to(device)

# Load trained parameters
state_dict = torch.load("resnet18_leafsnap.pth", map_location=device)
model.load_state_dict(state_dict)


model.eval()

dummy_input = torch.randn(1, 3, 224, 224, device=device)



onnx_path = "resnet18_leafsnap.onnx"

torch.onnx.export(
    model,                      # model being run
    dummy_input,                # model input
    onnx_path,                  # output file
    export_params=True,         # store trained parameters
    opset_version=11,           # stable + widely supported
    do_constant_folding=True,   # optimization
    input_names=["input"],      # input tensor name
    output_names=["output"],    # output tensor name
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

print("ONNX model saved successfully.")