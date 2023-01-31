# Manual Neural Network Implementation in C++

<p align="center">
  <img src="https://user-images.githubusercontent.com/113403062/215684930-c253936c-cf34-482f-abeb-58d7a32564e7.svg" alt="animated" width=700 heigt=700/>
</p>

This project was made from December 2022 to January 2023. In this project, I implemented a **Matrix** class and a **NeuralNetwork** class to simluate the behavior of a fully-connected neural network with one hidden layer. Provided above is the **heart.dat** file which I used to test my implementation. This file contains numerical data points relevant to the heart conditions of almost 300 patients. After standardization, the neural network achieved <10% mean squared error on predictions of the final column in this data file. 

**Matrix.cpp** is a C++ file that contains all of the code relevant to the matrix functionality of the neural network. This includes randomization, matrix operations, scaling, standardization, shuffling, dropping values, and memory allocation. This class can also be used in other C++ projects in which matrices are needed.

**NeuralNetwork.cpp** is a C++ file that contains all of the code of the neural network. Most importantly, this includes initializing the weights of the nodes, forward propogation, and backward propogation. There is also a function at the bottom of the file that allows one to obtain the predictions of their neural network after training. 

**Global.cpp** is a C++ file that contains all functions that I needed to use across **main.cpp, Matrix.cpp,** and **NeuralNetwork.cpp**. I am not sure if having a file like this is best practice as this is my first time ever coding in C++, but it was most intuitive for me. It includes the activation functions for the neural network and other functions relevant to data preparation. 

NOTE: This project is likely not complete. As you might have noticed, it is fairly difficult to use at the moment, considering the manual pre-processing required to format the data before feeding it to the neural network. I hope to fix this in the future. I also intend to add a second hidden layer, as well as a visualization functionality.

