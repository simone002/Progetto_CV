# Progetto_CV
🎯 Project Goal
[cite_start]This project aims to develop an automatic classification system for images of airborne particles acquired via a microscope. [cite_start]The main objective is to distinguish between different types of pollen and other particles (debris), addressing challenges such as the high visual similarity between some classes and the variability of the images. [cite_start]To achieve this, Deep Learning architectures were used, specifically Convolutional Neural Networks (CNNs).   

🗃️ Dataset
[cite_start]The dataset, provided by Prof. Sebastiano Battiato, is divided into a training set (

TRAIN_CV) and a test set (TEST_CV).   

Structure and Classes
[cite_start]The training set (

TRAIN_CV) is organized into 4 folders, each representing a class:   

[cite_start]

Corylus avellana - well-developed: Samples of normal hazelnut pollen.   

[cite_start]

Corylus avellana - anomalous: Samples of anomalous hazelnut pollen.   

[cite_start]

Alnus - well-developed: Samples of alder pollen.   

[cite_start]

Debris: Non-pollen particles such as bubbles, dust, and other debris.   

[cite_start]Each class folder contains three types of representations:   

[cite_start]

train_OBJ: Real color images of the objects.   

[cite_start]

train_MASK: Binary masks that isolate the object from the background.   

[cite_start]

train_SEGM: Detailed segmentations outlining the object's borders.   

[cite_start]Exploratory data analysis revealed a class imbalance, with the 

Alnus class being the most represented.   

⚙️ Methodology and Implementation
A complete framework was developed for training and evaluating different CNN models for the 4-class classification task.

Tested Architectures
[cite_start]Three different architectures, pre-trained on ImageNet and subsequently fine-tuned for this specific task, were compared:   

[cite_start]

ResNet-34: A 34-layer residual architecture known for its effectiveness in addressing the vanishing gradient problem.   

[cite_start]

VGG: An architecture that uses small-sized (3×3) convolutional filters.   

[cite_start]

AlexNet: One of the pioneering CNN architectures that demonstrated the effectiveness of deep learning for image classification.   

Preprocessing and Data Augmentation
The preprocessing pipeline was a crucial step for the project's success and includes:

[cite_start]

Mask Application: The original images (OBJ) are multiplied element-wise with the binary masks (MASK) to isolate the particles and remove background noise.   

[cite_start]

Resize: All images were resized to 128×128 pixels.   

[cite_start]

Normalization: Pixel values were normalized.   

[cite_start]To increase model robustness, 

data augmentation was applied during training, including random rotations, horizontal flips, and color jittering.   

Training Configuration
[cite_start]All models were trained with the following hyperparameters for a fair comparison:   

[cite_start]

Optimizer: Adam    

[cite_start]

Learning Rate: 1e-4    

[cite_start]

Loss Function: Cross Entropy Loss    

[cite_start]

Batch Size: 32    

[cite_start]

Maximum Epochs: 50, with an Early Stopping mechanism to prevent overfitting.   

[cite_start]A 

weighted random sampling strategy was used to manage class imbalance.   

Libraries and Frameworks
The implementation was developed using Python and the following main libraries:

[cite_start]

PyTorch and Torchvision for deep learning and pre-trained models.   

[cite_start]

Scikit-learn for evaluation metrics.   

[cite_start]

TensorBoard for training monitoring.   

[cite_start]

Matplotlib/Seaborn for results visualization.   

📊 Results
[cite_start]All tested models achieved excellent performance, exceeding 

96% accuracy on the test set.   

Model Comparison
[cite_start]

ResNet-34 proved to be the superior model, offering the best trade-off between accuracy and computational efficiency.   

Model	Accuracy	Macro Avg F1	Parameters
ResNet-34	[cite_start]	
0.9849    

[cite_start]	
0.98    

[cite_start]	
21M    

VGG	
[cite_start]0.9725    

[cite_start]0.97    

[cite_start]138M    

AlexNet	
[cite_start]0.9646    

[cite_start]0.96    

[cite_start]61M    

[cite_start]The analysis showed that the 

Debris class was classified almost perfectly by all models [cite_start], while the most common confusion occurred between the    

Corylus_normal and Alnus classes due to their morphological similarity.   

💡 Conclusions and Future Work
[cite_start]The project has successfully demonstrated the effectiveness of deep learning for automatic pollen classification. [cite_start]The preprocessing approach using binary masks proved to be crucial for the high performance achieved.   

Limitations
[cite_start]The dataset is limited to a few species and was collected under controlled conditions.   

[cite_start]The approach relies on the availability of pre-generated segmentation masks
