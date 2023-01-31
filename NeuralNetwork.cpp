#include "Matrix.h"
#include "NeuralNetwork.h"
#include "Global.h"

NeuralNetwork::NeuralNetwork(const vector<int> layers, const double learningrate, const int iterations) :
    Layers(layers), 
    LearningRate(learningrate),
    Iterations(iterations) {;}

NeuralNetwork::NeuralNetwork() : LearningRate(-1), Iterations(-1) {;}

NeuralNetwork::~NeuralNetwork() {;}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& nn) : LearningRate(nn.getLearningRate()), Iterations(nn.getIterations()) {
    vector<int> layers;
    for (int layer : nn.getLayers()) {
        layers.push_back(layer);
    }
    Layers = layers;
}

NeuralNetwork NeuralNetwork::operator=(const NeuralNetwork nn) {
    return nn;
}

// GETTER METHODS -----------------------------------------------

vector<double> NeuralNetwork::getLosses() const {
    return Loss;
}

double NeuralNetwork::getLearningRate() const {
    return LearningRate;
}

int NeuralNetwork::getIterations() const {
    return Iterations;
}

vector<int> NeuralNetwork::getLayers() const {
    return Layers;
}

// OTHER METHODS ------------------------------------------------

void NeuralNetwork::initWeights() {
    srand(1);
    Matrix W1 = Matrix(Layers[0], Layers[1]);
    Matrix b1 = Matrix(1, Layers[1]);
    Matrix W2 = Matrix(Layers[1], Layers[2]);
    Matrix b2 = Matrix(1, Layers[2]);

    W1.randomizeMatrix();
    b1.randomizeMatrix();
    W2.randomizeMatrix();
    b2.randomizeMatrix();

    Params.insert(make_pair("W1", W1));
    Params.insert(make_pair("b1", b1));
    Params.insert(make_pair("W2", W2));
    Params.insert(make_pair("b2", b2));
}

pair<double, Matrix> NeuralNetwork::forwardPropogation() {

    Matrix W1 = Params.find("W1") -> second;
    Matrix b1 = Params.find("b1") -> second;
    Matrix Z1 = (X.dot(W1)).addRowWeight(b1);
    Matrix A1 = Z1.applyMatrixOperation(Global::relu);


    Matrix W2 = Params.find("W2") -> second;
    Matrix b2 = Params.find("b2") -> second;
    Matrix Z2 = (A1.dot(W2)).addRowWeight(b2);
    Matrix yHat = Z2.applyMatrixOperation(Global::sigmoid);

    double loss = Global::entropyLoss(y, yHat);

    // save calculated parameters
    Params.insert(make_pair("Z1", Z1));
    Params.insert(make_pair("A1", A1));
    Params.insert(make_pair("Z2", Z2));

    return make_pair(loss, yHat);
}

void NeuralNetwork::backPropogation(Matrix yHat) {

    Matrix yInv = y.applyMatrixOperation(Global::inverse);
    Matrix yHatInv = yHat.applyMatrixOperation(Global::inverse);

    Matrix firstTerm = Global::divideMatrices(yInv, yHatInv.applyMatrixOperation(Global::ETAF));
    Matrix secondTerm = Global::divideMatrices(y, yHat.applyMatrixOperation(Global::ETAF));
    Matrix dl_WRT_yHat = Global::subtractMatrices(firstTerm, secondTerm);

    Matrix dl_WRT_sig = Global::multiplyMatrices(yHat, yHatInv);
    Matrix dl_WRT_Z2 = Global::multiplyMatrices(dl_WRT_yHat, dl_WRT_sig);

    Matrix W2 = Params.find("W2") -> second;
    Matrix dl_WRT_A1 = dl_WRT_Z2.dot(W2.transpose());
    Matrix A1 = Params.find("A1") -> second;
    Matrix dl_WRT_W2 = (A1.transpose()).dot(dl_WRT_Z2);
    Matrix dl_WRT_b2 = Global::sumMatrix(dl_WRT_Z2, 0);

    Matrix Z1 = Params.find("Z1") -> second;
    Matrix dl_WRT_Z1 = Global::multiplyMatrices(dl_WRT_A1, Z1.applyMatrixOperation(Global::dRelu));
    Matrix dl_WRT_W1 = (X.transpose()).dot(dl_WRT_Z1);
    Matrix dl_WRT_b1 = Global::sumMatrix(dl_WRT_Z1, 0);

    auto searchW1 = Params.find("W1");
    searchW1 -> second = Global::subtractMatrices(searchW1 -> second, dl_WRT_W1.scaleBy(LearningRate));

    auto searchW2 = Params.find("W2");
    searchW2 -> second = Global::subtractMatrices(searchW2 -> second, dl_WRT_W2.scaleBy(LearningRate));

    auto searchb1 = Params.find("b1");
    searchb1 -> second = Global::subtractMatrices(searchb1 -> second, dl_WRT_b1.scaleBy(LearningRate));

    auto searchb2 = Params.find("b2");
    searchb2 -> second = Global::subtractMatrices(searchb2 -> second, dl_WRT_b2.scaleBy(LearningRate));
}

void NeuralNetwork::fit(Matrix initialX, Matrix initialY) {
    X = initialX;
    y = initialY;
    initWeights();

    for (int i = 0; i < Iterations; i++) {
        pair<double, Matrix> result = forwardPropogation();
        backPropogation(result.second);
        Loss.push_back(result.first);
    }   
}

Matrix NeuralNetwork::predict(Matrix X) {
    Matrix W1 = Params.find("W1") -> second;
    Matrix b1 = Params.find("b1") -> second;
    Matrix Z1 = (X.dot(W1)).addRowWeight(b1);
    Matrix A1 = Z1.applyMatrixOperation(Global::relu);

    Matrix W2 = Params.find("W2") -> second;
    Matrix b2 = Params.find("b2") -> second;
    Matrix Z2 = (A1.dot(W2)).addRowWeight(b2);
    Matrix prediction = Z2.applyMatrixOperation(Global::sigmoid);

    return prediction;
}

