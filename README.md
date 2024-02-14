
# Face Emotion Detection Using Deep Learning

## Introduction
My model uses a deep learning technique to detect human emotions from facial expressions in real-time using web cam. The model captures video from a webcam, processes it to detect faces, and classifies each face into one of seven emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise. 

## Requirements
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
Requirement.txt is attached with the code. 

## Installation
   
   Install all required libraries using pip:
   
   	    
        pip install -r requirements.txt


## Dataset
I have used Kaggle FER2013 Dataset which can be downloaded from this site https://www.kaggle.com/datasets/msambare/fer2013. I have also attached the archive file and the dataset_folder which is being used by the model. If downloaded the file from the site please unzip it using this command 
	
	import zipfile
	zip_file_path = 'path/archive.zip' path
	unzip_location = 'path/dataset_folder'

The dataset folder consists of 2 subfolder train and test. These both the file consist of 7 sub folder namely the emotions Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise. It have over 28000 images in train and 7000 images in test dataset.

## Usage
Run the "maincode.ipynb" file in your jupyter notebook. 
I have used Visual Studio for my code compilation. 
There is also a saved tensorflow model "face_model_savedmodel"  as a tensorflow file. This is the already saved model with all the extracted features. 

If you do not want to run the whole code with training and testing you can just run the "EmotionDetection.ipynb" in the jupyter notebook. This is the real time emotion detection code.  

## CNN Model Architecture for Face Emotion Detection

My face emotion detection model uses a Convolutional Neural Network (CNN) architecture which is designed to classify facial expressions(real time) into one of seven emotions. The model’s architecture is detailed below:

1. **Input Shape**: 
   - The Input images are of shape 48x48 pixels which are in gray scale. It helps the model to easily process the images and removes pre-processing steps for  training the model.

2. **Convolutional Layers**: 
   - This model uses 4 cnn layers of size 3*3 with 64, 126, 256 and 512 filters respectively. I used 4 layers because the accuracy of the model was increased after 4 layers. 

3. **Batch Normalization**: 
   - I have used batch normalisation after every CNN layer. Batch normalization standardizes the outputs of the previous layer which results in  faster training and improved overall performance of the model.

4. **ReLU Activation Function**: 
   - A rectified linear unit (ReLU) is an activation function that introduces the property of non-linearity to a deep learning model and solves the vanishing gradients issue. I have used it after avery cnn layer to increase the robustness of my model

5. **Max Pooling Layers**: 
   - The max pooling layer is a crucial component of a convolution neural network architecture, and it helps the neural network extract important features from the input while simultaneously reducing the dimensions of the data. 

6. **Flattening Layer MLP**: 
   - All the extracted feature maps are now flattened into a one-dimensional vector. This process is essential to transition from the cnn layers to fully connected layers.

7. **Dense Layer MLP**: 
   - The flattened output is fed into fully connected layers for further processing. The first dense layer has 1024 neurons, providing a high level of neural inter connectivity for processing the complex features extracted by the cnn layers.
   - The final layer of the model is also a dense layer, consisting of 7 neurons, each corresponding to one of the emotion categories (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise). It uses a softmax activation function to output a probability distribution over the 7 classes.

8. **Model Compilation**: 
   - The model is compiled with the Adam optimizer which is an extension to stochastic gradient descent. 
   - The loss function used is categorical cross-entropy, a standard choice for multi-class classification problems, as it measures the performance of the model whose output is a probability value between 0 and 1.

9. **Training and Evaluation**: 
   - The model is trained on the images from the FER 2013 dataset which are already if the size 48*48 and grey scaled..
   - Evaluation metrics include accuracy, the confusion matrix, and a detailed classification report that provides insights into the model's performance across the different emotion categories.

## Configuration

The model is trained on images of size 48x48 pixels in gray scale. 

## Contact

Paras Jain 
pj2196@g.rit.edu

## Acknowledgments

Thanks to Kaggle for the dataset FER2013.

Thanks to Professor Alm for this wonderful opportunity