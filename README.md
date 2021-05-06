# A Simple Machine Learning Alphabets Learning App using TensorFlow.js
 This app lets you to guess the alphabets based on the random images appearing on the screen and it recognizes your drawings using machine learning tools.
 
 This model includes the following steps.
 
 -> Training Data
 
  we are going to use the MNIST letters dataset which is a dataset of handwritten images of english alphabets. Its difficult to train the different images in the 
  web browser, so we transfer the images to sprite sheet(everything is combined into a single image). MNISTalphabets_data_extraction.ipynb file includes the sprite sheet generator using python.
  
 -> Machine Learning
 
 We created a convolutional neural network that can classify the images of alphabets in the MNIST letters dataset. The CNN has conv2d, maxPooling2d,
 flatten, and dense layers. Since the MNIST letters has 26 letters, output layer should have 26 units and a softmax function.
 We are saving the tensor flow model as a json file.
 
 -> Predict the output
 
  Using the html canvas , we can draw images and predict the output based on the drawing.
  
 

