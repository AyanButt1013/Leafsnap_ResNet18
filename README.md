# Leafsnap Leaf Classification Using ResNet18



## Project Overview

Leafsnap is a leaf classification project that uses deep learning to automatically identify tree species from leaf images. Our initial custom CNN architecture plateaued at ~63% validation accuracy. To improve performance, we applied **transfer learning using ResNet18 pretrained on ImageNet**.  

The model leverages features learned from a large, diverse dataset and fine-tunes them for the **Leafsnap Field dataset**, which contains **185 tree species**.

---

## Features

- Upload leaf images for classification
- Predict tree species using a pretrained ResNet18 model
- REST API built with **FastAPI**
- ONNX model support for deployment
- Clean preprocessing pipeline using PyTorch transforms

---

## Dataset

The project uses the **Leafsnap Field Dataset**:

- Contains **185 tree species**
- Images resized to **224x224**
- Preprocessing includes normalization and conversion to tensor

## Technology Stack

- Python 3.9
- PyTorch
- ONNX
- FastAPI for API deployment
- Jupyter Notebook for experimentation
- VS Code / Command Line for development

