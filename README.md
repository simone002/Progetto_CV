# Progetto_CV

## 🎯 Project Goal
This project aims to develop an automatic classification system for images of airborne particles acquired via a microscope. The main objective is to distinguish between different types of pollen and other particles (debris), addressing challenges such as the high visual similarity between some classes and the variability of the images. To achieve this, Deep Learning architectures were used, specifically Convolutional Neural Networks (CNNs).

## 🗃️ Dataset
The dataset, provided by Prof. Sebastiano Battiato, is divided into a training set (TRAIN_CV) and a test set (TEST_CV).

### Structure and Classes
The training set (TRAIN_CV) is organized into 4 folders, each representing a class:

- **Corylus avellana - well-developed**: Samples of normal hazelnut pollen
- **Corylus avellana - anomalous**: Samples of anomalous hazelnut pollen
- **Alnus - well-developed**: Samples of alder pollen
- **Debris**: Non-pollen particles such as bubbles, dust, and other debris

Each class folder contains three types of representations:

- **train_OBJ**: Real color images of the objects
- **train_MASK**: Binary masks that isolate the object from the background
- **train_SEGM**: Detailed segmentations outlining the object's borders

Exploratory data analysis revealed a class imbalance, with the Alnus class being the most represented.

## ⚙️ Methodology and Implementation
A complete framework was developed for training and evaluating different CNN models for the 4-class classification task.

### Tested Architectures
Three different architectures, pre-trained on ImageNet and subsequently fine-tuned for this specific task, were compared:

- **ResNet-34**: A 34-layer residual architecture known for its effectiveness in addressing the vanishing gradient problem
- **VGG**: An architecture that uses small-sized (3×3) convolutional filters
- **AlexNet**: One of the pioneering CNN architectures that demonstrated the effectiveness of deep learning for image classification

### Preprocessing and Data Augmentation
The preprocessing pipeline was a crucial step for the project's success and includes:

- **Mask Application**: The original images (OBJ) are multiplied element-wise with the binary masks (MASK) to isolate the particles and remove background noise
- **Resize**: All images were resized to 128×128 pixels
- **Normalization**: Pixel values were normalized

To increase model robustness, data augmentation was applied during training, including random rotations, horizontal flips, and color jittering.

### Training Configuration
All models were trained with the following hyperparameters for a fair comparison:

- **Optimizer**: Adam
- **Learning Rate**: 1e-4
- **Loss Function**: Cross Entropy Loss
- **Batch Size**: 32
- **Maximum Epochs**: 50, with an Early Stopping mechanism to prevent overfitting

A weighted random sampling strategy was used to manage class imbalance.

### Libraries and Frameworks
The implementation was developed using Python and the following main libraries:

- **PyTorch and Torchvision** for deep learning and pre-trained models
- **Scikit-learn** for evaluation metrics
- **TensorBoard** for training monitoring
- **Matplotlib/Seaborn** for results visualization

## 📊 Results
All tested models achieved excellent performance, exceeding 96% accuracy on the test set.

### Model Comparison
ResNet-34 proved to be the superior model, offering the best trade-off between accuracy and computational efficiency.

| Model     | Accuracy | Macro Avg F1 | Parameters |
|-----------|----------|--------------|------------|
| ResNet-34 | 0.9849   | 0.98         | 21M        |
| VGG       | 0.9725   | 0.97         | 138M       |
| AlexNet   | 0.9646   | 0.96         | 61M        |

The analysis showed that the Debris class was classified almost perfectly by all models, while the most common confusion occurred between the Corylus_normal and Alnus classes due to their morphological similarity.

## 💡 Conclusions and Future Work
The project has successfully demonstrated the effectiveness of deep learning for automatic pollen classification. The preprocessing approach using binary masks proved to be crucial for the high performance achieved.

### Limitations
- The dataset is limited to a few species and was collected under controlled conditions
- The approach relies on the availability of pre-generated segmentation masks
