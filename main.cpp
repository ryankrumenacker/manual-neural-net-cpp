#include <iostream>

#include "Matrix.h"
#include "NeuralNetwork.h"
#include "Global.h"

using namespace std;

int main() {

    Matrix heartDataframe = Global::generateMatrix("heart.dat");
    heartDataframe.shuffleRows();

    Matrix X = heartDataframe.dropColumn(heartDataframe.getCols() - 1);

    heartDataframe.replaceValueInColumn(heartDataframe.getCols() - 1, 1, 0);
    heartDataframe.replaceValueInColumn(heartDataframe.getCols() - 1, 2, 1);

    Matrix yLabels = heartDataframe.getMatrixColumn(heartDataframe.getCols() - 1);

    double trainSize = 0.75;
    vector<Matrix> result = Global::splitData(X, yLabels, trainSize);

    Matrix Xtrain = result[0];
    Matrix Ytrain = result[1];
    Matrix Xtest = result[2];
    Matrix Ytest = result[3];

    Xtrain.standardizeColumns();
    Xtest.standardizeColumns();

    int inputLayer = X.getCols();   
    int hiddenLayer = 8;
    int outputLayer = 1;

    vector<int> layers = {inputLayer, hiddenLayer, outputLayer};

    double learningRate = 0.001;
    int iterations = 1000;
    NeuralNetwork nn = NeuralNetwork(layers, learningRate, iterations);

    nn.fit(Xtrain, Ytrain);
    
    // for (double loss : nn.getLosses()) {
    //     std::cout << loss << std::endl;
    // }

    Matrix prediction = nn.predict(Xtest);

    double MSE = Global::getMeanSquaredError(prediction, Ytest);
    std::cout << MSE << std::endl;
};