#include <map>
#include <cstdlib>
#include <cmath>
#include <stdlib.h>
#include <functional>
#include <time.h>
#include <tgmath.h>
#include <algorithm> // max

#include "Matrix.h"

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

using namespace std;

class NeuralNetwork {
public:
    map<string, Matrix> Params;
    double LearningRate;
    int Iterations;
    vector<double> Loss;
    vector<int> Layers;
    Matrix X;
    Matrix y;

    // CONSTRUCTORS and DESTRUCTORS --------------------

    NeuralNetwork(const vector<int> layers, const double learningrate, const int iterations);

    NeuralNetwork();

    ~NeuralNetwork();

    NeuralNetwork(const NeuralNetwork&);

    NeuralNetwork operator=(const NeuralNetwork);

    // GETTER METHODS -------------------------------------

    vector<double> getLosses() const;

    double getLearningRate() const;

    int getIterations() const;

    vector<int> getLayers() const;

    // OTHER METHODS -------------------------------------

    void initWeights();

    pair<double, Matrix> forwardPropogation();

    void backPropogation(Matrix yHat);

    void fit(Matrix initialX, Matrix initialY);

    Matrix predict(Matrix X);


};

#endif