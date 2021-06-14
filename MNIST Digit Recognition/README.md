# MNIST Digit Recognition

The famous MNIST handwritten digit recognizer using deep neural networks utilizing TensorFlow as the framework. Input an image and let the deep neural network recognize digits from it. The network architecture consists of 3 Convolutional layers using 3x3 filters each, followed by MaxPooling on each convolutional layer and then 2 fully connected or Dense layers with 10 unique output classes for each unique digit.

The application is live [here](http://the-ml-dl-app.herokuapp.com/deep-learning/digit-recognition)

## Dataset
- The dataset used is MNIST as jpg from Kaggle
- [Dataset Link](https://www.kaggle.com/scolianni/mnistasjpg)

## Technologies Used
![Python](https://img.shields.io/badge/-Python-FFFFFF?style=flat&logo=python&logoColor=3776AB)&nbsp;&nbsp;&nbsp;
![Numpy](https://img.shields.io/badge/-NumPy-FFFFFF?style=flat&logo=numpy&logoColor=013243)&nbsp;&nbsp;&nbsp;
![TensorFlow](https://img.shields.io/badge/-TensorFlow-FFFFFF?style=flat&logo=tensorflow&logoColor=FF6F00)&nbsp;&nbsp;&nbsp;
![Colaboratory](https://img.shields.io/badge/-Google%20Colab-FFFFFF?style=flat&logo=google-colab&logoColor=F9AB00)

## Installations Required
For running the source code on your local machine, the following dependencies are required.
- Python
- Numpy
- TensorFlow

## Launch
For local installation, follow these steps:
1. Download source code from this repository or click [here](https://github.com/rishabh1323/Deep-Learning-Basic-Projects/archive/refs/heads/main.zip).
2. Extract files to desired directory.
3. [Download](https://www.python.org/downloads/) and install Python3 if not done already.
4. Create a new python virtual environment.
```
python3 -m venv tutorial-env
```
5. Once you’ve created a virtual environment, you may activate it.  

`On Windows, run:`
```
tutorial-env\Scripts\activate.bat
```
`On Unix or MacOS, run:`
```
source tutorial-env/bin/activate
```
> Refer to [python documentation](https://docs.python.org/3/tutorial/venv.html) for more information on virtual environments.  
6. Install the required dependecies.
```
pip install numpy pandas tensorflow
```
7. Launch the Jupyter Notebooks or run the python files now.
