
## Pneumonia Detection Model
This project aims to classify chest X-ray images into two categories: pneumonia and normal using a deep learning model. The VGG16 pre-trained model is utilized with transfer learning to perform the classification, leveraging the Kaggle dataset "Chest X-Ray Images (Pneumonia)".
## Project Overview
The project involves creating a deep learning classifier using chest X-ray images to differentiate between pneumonia and normal cases. The VGG16 pre-trained model was fine-tuned for this classification task. The model's performance was evaluated on a test dataset, and results were analyzed using various metrics.
## Dataset
Name of the dataset is Chest X-Ray Images (Pneumonia) by Paul Mooney. The dataset used in this projec is obtained from Kaggle and includes the following components:  

**Training Data:** Located in the chest_xray/train directory, containing images labeled as NORMAL or PNEUMONIA.  
**Validation Data:** A subset of the training data used for validation purposes, located in the chest_xray/val directory.  
**Test Data:** Used to evaluate the model's performance, located in the chest_xray/test directory.
## Installation and Requirements
This project requires the following Python libraries:

+ TensorFlow 2.16.1
+ NumPy
+ Pandas
+ Matplotlib
+ Seaborn
+ Scikit-learn
+ OpenCV
+ PIL
## Model Architecture
The model architecture used in this project is as follows:

+ **VGG16:** A pre-trained VGG16 model was used as the base model for transfer learning. The top layers of VGG16 were removed, and a classification head was added.  
+ **GlobalAveragePooling2D:** This layer summarizes the output of VGG16, reducing its dimensionality.  
+ **Dense:** The final layer uses a sigmoid activation function to output the probabilities for the two classes (NORMAL and PNEUMONIA).
## Training Process
The model was trained with the following hyperparameters:

+ **Optimizer:** Adam  
+ **Loss Function:** Binary Cross-Entropy  
+ **Metrics:** Accuracy  
+ **Epochs:** 100  
+ **Batch Size:** 16  
## Data Augmentation
Data augmentation techniques applied during training include:  
* **Rotation Range:** Random rotation of images up to 40 degrees
* **Width Shift:** Horizontal shift of images by up to 20%
* **Height Shift:** Vertical shift of images by up to 20%
* **Shear Range:** Random shear transformations
* **Zoom Range:** Zooming in or out by up to 20%
* **Horizontal Flip:** Random horizontal flipping of images
## Model Training
This project leverages transfer learning to build a binary image classifier. I opted to utilize a pre-trained VGG16 model, freezing its weights obtained from training on the ImageNet dataset. This approach allows the model to benefit from the rich feature representations learned by VGG16 on a massive dataset, while focusing on adapting to the specific task at hand.  
The architecture consists of the VGG16 base model (excluding the top classification layers), followed by a Global Average Pooling layer to condense the spatial information into a feature vector. Finally, a single dense layer with a sigmoid activation function is used to output the binary classification probability.  
I chose the Adam optimizer for its efficiency in handling sparse gradients and its generally good performance on image classification tasks. Binary cross-entropy loss was selected as the appropriate loss function for this binary classification problem.  
The model was trained for 100 epochs, with the training process monitored through the evaluation of accuracy on both the training and validation sets. This allowed me to track the model's progress and identify potential issues like overfitting. The number of steps per epoch was determined by the number of samples in each dataset divided by the batch size used during training.  
By leveraging transfer learning and carefully selecting the model architecture and training parameters, I aimed to build a robust and accurate binary image classifier.

I acknowledge the zigzags in the graph. These are mostly caused by the learning rate. A smaller learning rate would have prevented this. However, I didn't have much time to run the model since my TPU time left on the kaggle was really low.
