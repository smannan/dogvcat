# Abstract 
This project was designed to classify images of dogs and cats. The two methods implemented were Random Forests 
using scikit-learn and Convolutional Nueral Networks using Keras. The documentation for scikit-learn random
forests can be found here: http://scikit-learn.org/stable/ and for Keras: https://keras.io/

# Dataset
The dataset for this project was downloaded from Kaggle and can be found here: 
https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition. 
The dataset set consists of 25,000 images, 12,500 dogs and 12,500 cats.

# Import and Tidy
This notebook imports the images and processses them for scikit-learn and Keras. To classify the images in scikit-learn the images are saved and loaded as numpy arrays. The images are first cropped to a uniform size, converted to greyscale, and saved. Next, Histogram of Oriented Gradients (HOG) and Principal Component Analysis (PCA) were used on the numpy arrays to extract key features and reduce the dimensions of each image. These grey scaled images, HOG, and PCA features were all saved as numpy arrays to be used later. 

For Keras, the images were split in training and validation sets. 70% of the images were used for training and 30% were used for validation. The images were initially downloaded into one folder. Then, 30% of the images were randomly selected, seperated by dogs and cats, and moved into a validation directory. The rest of the images were moved into a train directory with the same structure. See the Import and Tidy notebook to view the directory structure used to train Keras models.

# Exploratory Data Analysis (EDA)
This notebook visualizes the processed images and HOG features and used Principal Component Analysis (PCA) to analyze the relationship between cat and dog images in two and three dimensions. 

# Modeling Forest
This notebook uses HOG and PCA features on a random forest classifier to come up with baseline accuracy on the predictions. A grid search is used to find the optimal model and a five-fold cross validation is plotted to measure the performance. An averaged log loss and binary accuracy are also computed to measure performance. Both feature sets resulted in similar performances: a log loss around 0.69 and binary accuracy of 71%.

# Convolutional Neural Networks (CNN)
This notebook implements a CNN to predict dog versus cat images. Keras was used to construct, train, and evaluate the CNN. Before training and testing, the images were first processed. They were rotated and flipped in various angles to augment the dataset and improve performance. Images were then fed into the network for training via a generator as 150x150 RGB pixel vectors. Using a generator to train the model saved a significant amount of space. Finally, the model was evaluated on a testing set (30% of the images were withheld during training). The CNN trained on 50 epochs with a batch size of 16 and RMSprop optimizer with the default learning rate of 0.01. The CNN was able to reach 77% validation accuracy with a log loss around 0.49. 

# Works Cited
1. https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/
2. https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
3. http://machinelearningmastery.com/
4. https://keras.io/
5. http://scikit-learn.org/stable/documentation.html
6. https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition
7. https://www.youtube.com/user/sentdex
