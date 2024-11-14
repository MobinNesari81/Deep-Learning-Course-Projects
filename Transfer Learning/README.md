
# 🖼️ Image Classification with ResNeXt

This project demonstrates the use of a **ResNeXt** model for image classification 
using the **Intel Image Classification dataset**. ResNeXt is a deep convolutional neural network
architecture known for its efficiency and high performance in image classification tasks.

## 📚 Project Overview

In this notebook, we will:
1. **Load and Preprocess** the Intel Image Classification dataset.
2. **Build a ResNeXt Model** with customized configurations for this task.
3. **Train the ResNeXt Model** on the training set and evaluate on the test set.
4. **Analyze Results** by visualizing accuracy and loss trends.

ResNeXt uses a modular structure with grouped convolutions, allowing it to achieve high accuracy with fewer parameters compared to other deep models.

## 🛠️ Setup Instructions

To get started, follow these setup steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/ResNeXt_Classification.git
   cd ResNeXt_Classification
   ```

2. **Install dependencies** - this notebook requires PyTorch, NumPy, Pandas, and Matplotlib:
   ```bash
   pip install torch torchvision numpy pandas matplotlib
   ```

3. **Run the Notebook** - Open the notebook using Jupyter Notebook or Jupyter Lab:
   ```bash
   jupyter notebook resnext_classification.ipynb
   ```

## 🚀 Usage

Each cell in the notebook is documented with explanations and code comments. 
Simply follow the cells in order for a step-by-step experience building and evaluating a ResNeXt model.

### Key Sections and Code Examples:

1. **Importing Libraries and Setting Up the Device**: We import the necessary libraries and configure the device (GPU/CPU).

   ```python
   import torch
   import torch.nn as nn
   import torchvision.transforms as tt
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   ```

2. **Data Loading & Transformation**: We load the Intel Image Classification dataset and apply transformations like resizing, random cropping, and normalization.

   ```python
   train_transform = tt.Compose([
       tt.Resize(150),
       tt.RandomCrop(150),
       tt.RandomHorizontalFlip(),
       tt.ToTensor(),
       tt.Normalize((0.4951, 0.4982, 0.4979), (0.2482, 0.2467, 0.2807))
   ])
   ```

3. **ResNeXt Model Architecture**: The ResNeXt model is defined with layers configured for grouped convolutions.

   ```python
   class ResNeXt(nn.Module):
       def __init__(self):
           # Define ResNeXt model here
   ```

4. **Training the Model**: Using CrossEntropy loss and an optimizer, we train the model over multiple epochs.

   ```python
   loss = criterion(output, labels)
   loss.backward()
   optimizer.step()
   ```

5. **Evaluating and Visualizing Results**: After training, we evaluate the model and plot accuracy and loss trends.

## 🤔 Potential Improvements

- Experiment with different model depths and widths to achieve better results.
- Use data augmentation techniques like rotation and color jitter to improve generalization.
- Implement a learning rate scheduler for potentially faster convergence.

## 📬 Contact

For questions or suggestions, reach out via the Issues section of the repository or contact <mobinnesari81@gmail.com>.

Enjoy classifying with ResNeXt! ✨
