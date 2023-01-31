#include <iostream>
#include <fstream>

#include "Matrix.h"
#include "NeuralNetwork.h"
#include "Global.h"

using namespace std;

const double Global::PI = 3.14159265358979323846;
const double Global::E = 2.718281828459045235360;
const double Global::ETA = 0.0000000000001;

// applies the ReLU function to a double value
double Global::relu(double x) {
    return max(0.0, x);
}

// applies the Sigmoid function to a double value
double Global::sigmoid(double x) {
    return 1.0 / (1.0 + pow(E, -x)); 
}

// applies the derivative the ReLU function to a double value
double Global::dRelu(double x) {
    if (x > 0) {
        return 1;
    } return 0;
}

// prevents a value of 0 from begin passed into the logarithm function, which results in infinity
double Global::ETAF(double x) {
    return max(x, ETA);
}

double Global::inverse(double x) {
    return 1 - x;
}

// prints a Matrix  
void Global::printMatrix(Matrix mat) {
    for (int i = 0; i < mat.getRows(); i++) {
        for (int j = 0; j < mat.getCols(); j++) {
            cout << mat.getMatrixValue(i, j) << " ";
        }
        cout << "\n"; 
    }
}

// returns the sum of values in a Matrix    
double Global::sumMatrix(Matrix mat) {
    double sum = 0;
    for (int i = 0; i < mat.getRows(); i++) {
        for (int j = 0; j < mat.getCols(); j++) {
            sum += mat.getMatrixValue(i, j);
        }
    }
    return sum;
}

Matrix Global::sumMatrix(Matrix mat, int axis) {
    
    int rows;
    int cols;
    Matrix newMat;
    if (axis == 0) {
        rows = 1;
        cols = mat.getCols();
        newMat = Matrix(rows, cols);

        for (int i = 0; i < mat.getRows(); i++) {
            for (int j = 0; j < mat.getCols(); j++) {
                newMat.setMatrixValue(0, j, mat.getMatrixValue(i, j));
            }
        }

        return newMat;

    } else if (axis == 1) {
        rows = mat.getRows();
        cols = 1;
        newMat = Matrix(rows, cols);

        double currSum;
        for (int i = 0; i < mat.getRows(); i++) {
            currSum = 0;
            for (int j = 0; j < mat.getCols(); j++) {
                currSum += mat.getMatrixValue(i, j);
            }
            newMat.setMatrixValue(i, 0, currSum);
        }

        return newMat;

    } else {
        throw invalid_argument("invalid axis argument (sumMatrix)");
    }
}

Matrix Global::combination(Matrix mat1, Matrix mat2, double (*func)(double, double)) {
    if (mat1.getRows() != mat2.getRows() || mat1.getCols() != mat2.getCols()) {
        throw invalid_argument("matrices are not of appropriate dimensions for combination (combination)");
    }

    int rows = mat1.getRows();
    int cols = mat1.getCols();
    Matrix newMat = Matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            newMat.setMatrixValue(i, j, func(mat1.getMatrixValue(i, j), mat2.getMatrixValue(i, j)));
        }
    }

    return newMat;
}

Matrix Global::addMatrices(Matrix mat1, Matrix mat2) {
    auto addition = [](double x, double y) {
        return x + y;
    };  
    
    return combination(mat1, mat2, addition);
}

Matrix Global::subtractMatrices(Matrix mat1, Matrix mat2) {
    auto subtraction = [](double x, double y) {
        return x - y;
    };

    return combination(mat1, mat2, subtraction);
}

Matrix Global::multiplyMatrices(Matrix mat1, Matrix mat2) {
    auto multiplication = [](double x, double y) {
        return x * y;
    };  
    
    return combination(mat1, mat2, multiplication);
}

Matrix Global::divideMatrices(Matrix mat1, Matrix mat2) {
    auto division = [](double x, double y) {
        return x / y;
    };  
    
    return combination(mat1, mat2, division);
}

double Global::entropyLoss(Matrix y, Matrix yHat) {

    double loss;
    int sampleSize = y.getRows();

    Matrix yInv = y.applyMatrixOperation(inverse);
    Matrix yHatInv = yHat.applyMatrixOperationsInOrder(inverse, ETAF);
    Matrix yMat = y.applyMatrixOperation(ETAF);

    Matrix firstTerm = multiplyMatrices(yHat.applyMatrixOperation(log), y);
    Matrix secondTerm = multiplyMatrices(yHatInv.applyMatrixOperation(log), yInv);

    loss = (-1.0 / sampleSize) * sumMatrix(addMatrices(firstTerm, secondTerm));

    return loss;
}

Matrix Global::generateMatrix(string filePath) {

    ifstream file(filePath);
    vector<vector<double>> mat;
    vector<double> row;
    string currLine;
    string currString = "";
    if (file.is_open()) {
        while (getline(file, currLine)) {
            for (char c : currLine) {
                if (c == ' ') {
                    row.push_back(stod(currString));
                    currString = "";
                } else {
                    currString += c;
                }
            }
            if (!(currString.empty())) {
                row.push_back(stod(currString));
                currString = "";
            }
            mat.push_back(row);
            row.clear();
        }
    }

    file.close();

    // ensuring that all matrices entries have the same number of columns
    double rows = mat.size();
    double cols = -1;
    for (int i = 0; i < rows; i++) {
        if (cols == -1 || mat[i].size() == cols) {
            cols = mat[i].size();
        } else {
            throw invalid_argument("rows are not of the same size (generateMatrix)");
        }
    }

    Matrix result = Matrix(rows, cols, mat);

    return result;
}

// Xtrain = training data ((trainSize * 100)% of total data)
// Xtest = testing data ((1 - trainSize * 100)% of total data)
// Ytrain = answer key of Xtrain data
// Ytest = answer key of Xtest data
vector<Matrix> Global::splitData(Matrix X, Matrix y, double trainSize) {

    if (X.getRows() != y.getRows()) {
        throw invalid_argument("matrices are not of correct size (splitData)");
    }

    if (trainSize > 1 || trainSize < 0) {
        throw invalid_argument("training size is not between expected values (splitData)");
    }

    vector<Matrix> result;
    int totalRows = X.getRows();
    int breakPoint = (int) (totalRows * trainSize);

    Matrix Xtrain = Matrix(breakPoint, X.getCols());
    Matrix Ytrain = Matrix(breakPoint, y.getCols());

    for (int i = 0; i < breakPoint; i++) {
        Xtrain.setMatrixRow(i, X.getMatrixRow(i));
        Ytrain.setMatrixRow(i, y.getMatrixRow(i));
    }

    Matrix Xtest = Matrix(X.getRows() - breakPoint, X.getCols());
    Matrix Ytest = Matrix(X.getRows() - breakPoint, y.getCols());

    for (int i = breakPoint; i < totalRows; i++) {
        Xtest.setMatrixRow(i - breakPoint, X.getMatrixRow(i));
        Ytest.setMatrixRow(i - breakPoint, y.getMatrixRow(i));
    }

    result = { Xtrain , Ytrain , Xtest , Ytest };

    return result;
}

double Global::getMeanSquaredError(Matrix X, Matrix Y) {
    if (X.getRows() != Y.getRows() || X.getCols() != Y.getCols()) {
        throw invalid_argument("matrices are not of the same dimensions (getMeanSquaredError)");
    }

    int rows = X.getRows();
    int cols = X.getCols();

    if (cols != 1) {
        throw invalid_argument("matrices have more than one column of data (getMeanSquaredError)");
    }

    double MSE = 0;
    for (int i  = 0; i < rows; i++) {
        MSE += pow(X.getMatrixValue(i, 0) - Y.getMatrixValue(i, 0), 2);
    }

    return MSE / rows;
}

