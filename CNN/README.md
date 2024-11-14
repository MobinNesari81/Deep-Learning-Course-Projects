
# 🖼️ CIFAR-10 Classification with VGG16 and VGG18 Models

This project demonstrates the use of **VGG16** and **VGG18** models to classify images in the **CIFAR-10** dataset.
The VGG architecture, known for its simple and effective deep convolutional network structure, is a popular choice for 
image classification tasks.

## 📚 Project Overview

In this notebook, we will:
1. **Load and Preprocess** the CIFAR-10 dataset, applying data augmentations and normalization.
2. **Build and Train VGG16 and VGG18 Models** tailored to classify CIFAR-10 images.
3. **Evaluate Model Performance** using test accuracy and confusion matrices.

With the classification matrix, we can analyze the model's performance for each CIFAR-10 class individually.

## 🛠️ Setup Instructions

Follow these steps to get started with the notebook:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/CIFAR10_VGG_Classification.git
   cd CIFAR10_VGG_Classification
   ```

2. **Install dependencies** - this notebook requires PyTorch, NumPy, Matplotlib, and Seaborn:
   ```bash
   pip install torch torchvision numpy matplotlib seaborn
   ```

3. **Run the Notebook** - Open the notebook file in Jupyter Notebook or Jupyter Lab:
   ```bash
   jupyter notebook CIFAR10_VGG_Classification.ipynb
   ```

## 🚀 Usage

Each cell in the notebook includes code comments and markdown explanations for ease of understanding.
Simply follow the cells in order to train and evaluate the VGG models on CIFAR-10.

### Key Sections and Code Examples:

1. **Data Loading and Transformation**: The CIFAR-10 dataset is loaded and transformations are applied.

   ```python
   transform_train = transforms.Compose([
       transforms.RandomCrop(32, padding=4),
       transforms.RandomHorizontalFlip(),
       transforms.ToTensor(),
       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
   ])
   ```

2. **VGG Model Architecture**: The VGG model is defined with either 16 or 18 layers, based on the chosen configuration.

   ```python
   class VGGNet(nn.Module):
       def __init__(self, vgg_type, num_classes=10):
           # Define VGG layers here
   ```

3. **Training Loop**: The model is trained using CrossEntropy loss and an optimizer.

   ```python
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()
   ```

4. **Confusion Matrix Visualization**: After training, a confusion matrix is used to visualize predictions.

## 🤔 Potential Improvements

- Try adjusting the learning rate and batch size to see if accuracy improves.
- Experiment with additional layers or different activation functions.
- Apply data augmentations like color jitter or rotation to improve generalization.

## 📬 Contact

For questions or feedback, reach out via the repository's Issues section or contact <mobinnesari81@gmail.com>.

Enjoy exploring CIFAR-10 with VGG models! 🖼️✨
