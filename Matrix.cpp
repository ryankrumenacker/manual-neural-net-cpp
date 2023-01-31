#include "Matrix.h"
#include "NeuralNetwork.h"
#include "Global.h"

using namespace std;

// CONSTRUCTORS ------------------------------------------

Matrix::Matrix(const int rows, const int cols) : Rows(rows), Cols(cols) {
    allocateMemory();
    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j < Cols; j++) {
            pMat[i][j] = 0;
        }
    }
}

Matrix::Matrix(const int rows, const int cols, const vector<vector<double>> mat) : Rows(rows), Cols(cols) {
    allocateMemory();
    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j < Cols; j++) {
            pMat[i][j] = mat[i][j];
        }
    }
}

Matrix::Matrix() : Rows(-1), Cols(-1) {;}

Matrix::~Matrix() {
    for (int i = 0; i < Rows; ++i) {
        delete[] pMat[i];
    }
    delete[] pMat;
}

Matrix::Matrix(const Matrix& mat) : Rows(mat.getRows()), Cols(mat.getCols()) {
    allocateMemory();
    for (int i = 0; i < Rows; ++i) {
        for (int j = 0; j < Cols; ++j) {
            pMat[i][j] = mat.getMatrixValue(i, j);
        }
    }
}

Matrix& Matrix::operator=(const Matrix& mat) {
    if (this == &mat) {
        return *this;
    }

    if ((Rows != mat.getRows() || Cols != mat.getCols()) && Rows != -1 && Cols != -1) {
        for (int i = 0; i < Rows; ++i) {
            delete[] pMat[i];
        }
        delete[] pMat;
    }

    Rows = mat.getRows();
    Cols = mat.getCols();
    allocateMemory();

    for (int i = 0; i < Rows; ++i) {
        for (int j = 0; j < Cols; ++j) {
            pMat[i][j] = mat.getMatrixValue(i, j);
        }
    }

    return *this;
}

// GETTER METHODS --------------------------

int Matrix::getRows() const {
    if (Rows == -1) {
        throw invalid_argument("rows have not yet been initialized (getRows)");
    }
    return Rows;
}

int Matrix::getCols() const {
    if (Cols == -1) {
        throw invalid_argument("columns have not yet been initialized (getCols)");
    }
    return Cols;
}

double Matrix::getMatrixValue(int i, int j) const {
    if (i < 0 || j < 0 || i >= Rows || j >= Cols) {
        throw invalid_argument("matrix indices are out of bounds (getMatrixValue)");
    }

    return pMat[i][j];
}

Matrix Matrix::getMatrixRow(int rowIndex) const {
    if (rowIndex < 0 || rowIndex >= Rows) {
        throw invalid_argument("row to be removed is out of range (getMatrixRow)");
    }

    Matrix row = Matrix(1, Cols);
    for (int j = 0; j < Cols; j++) {
        row.setMatrixValue(0, j, pMat[rowIndex][j]);
    }

    return row;
}

Matrix Matrix::getMatrixColumn(int colIndex) const {
    if (colIndex < 0 || colIndex >= Cols) {
        throw invalid_argument("column to be removed is out of range (getMatrixColumn)");
    }

    Matrix column = Matrix(Rows, 1);
    for (int i = 0; i < Rows; i++) {
        column.setMatrixValue(i, 0, pMat[i][colIndex]);
    }

    return column;
}

// SETTER METHODS ---------------------------

void Matrix::setMatrix(Matrix mat) {
    if (mat.getRows() != Rows || mat.getCols() != Cols) {
        throw invalid_argument("matrices are not of the same dimensions (setMatrix)");
    }   

    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j < Cols; j++) {
            pMat[i][j] = mat.pMat[i][j];
        }
    }
}

void Matrix::setMatrixValue(int i, int j, double value) {
    if (i < 0 || j < 0 || i >= Rows || j >= Cols) {
        throw invalid_argument("matrix indices are out of bounds (setMatrixValue)");
    }

    pMat[i][j] = value;
}

void Matrix::setMatrixRow(int rowIndex, Matrix row) {
    if (row.getRows() != 1 || row.getCols() != Cols) {
        throw invalid_argument("row size is not of adequate dimensions (setMatrixRow)");
    }

    for (int j = 0; j < Cols; j++) {
        pMat[rowIndex][j] = row.getMatrixValue(0, j);
    }
}

// OTHER METHODS -------------------------------------------

// initializes random 2D matrix following a uniform normal distribution (using the Box-Muller method)
void Matrix::randomizeMatrix() {
    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j < Cols; j++) {
            pMat[i][j] = boxMuller();
        }
    }
}

// returns a Matrix with the given function applied to each value in the Matrix on which it is called
Matrix Matrix::applyMatrixOperation(double (*func)(double)) {
    Matrix newMat = Matrix(Rows, Cols);
    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j < Cols; j++) {
            newMat.pMat[i][j] = func(pMat[i][j]);
        }
    }
    return newMat;
}  

Matrix Matrix::applyMatrixOperationsInOrder(double (*func1)(double), double (*func2)(double)) {
    Matrix newMat = Matrix(Rows, Cols);
    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j < Cols; j++) {
            newMat.pMat[i][j] = func2(func1(pMat[i][j]));
        }
    }
    return newMat;
}  

Matrix Matrix::scaleBy(double scaleValue) {
    Matrix newMat = Matrix(Rows, Cols);
    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j < Cols; j++) {
            newMat.pMat[i][j] = pMat[i][j] * scaleValue;
        }
    }
    return newMat;
}

// returns dot product / matrix multiplication of two Matrices
Matrix Matrix::dot(Matrix mat) {

    if (Cols != mat.getRows()) {
        throw invalid_argument("matrices are not of appropriate dimensions for multiplication (dot)");
    }

    Matrix newMat = Matrix(Rows, mat.getCols());
    double currValue;
    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j < mat.getCols(); j++) {
            for (int k = 0; k < mat.getRows(); k++) {
                currValue = newMat.getMatrixValue(i, j);
                newMat.setMatrixValue(i, j, currValue + pMat[i][k] * mat.getMatrixValue(k, j));
            }
        }
    }

    return newMat;
}

Matrix Matrix::transpose() {
    Matrix newMat = Matrix(Cols, Rows);

    for (int i = 0; i < Cols; i++) {
        for (int j = 0; j < Rows; j++) {
            newMat.pMat[i][j] = pMat[j][i];
        }
    }

    return newMat;
}

void Matrix::standardizeColumns() {
    vector<double> means(Cols);
    
    // counting values in each column
    for (int i  = 0; i < Rows; i++) {
        for (int j = 0; j < Cols; j++) {
            means[j] += pMat[i][j];
        }
    }
    // dividing columns by number of rows to find means
    for (int i = 0; i < Cols; i++) {
        means[i] /= Rows;
    }

    // calculating the standard deviations of each column
    vector<double> deviations(Cols);
    for (int i  = 0; i < Rows; i++) {
        for (int j = 0; j < Cols; j++) {
            deviations[j] += ((pMat[i][j] - means[j]) * (pMat[i][j] - means[j]));
        }
    }
    // dividng each column by the number of rows and taking the square root
    for (int i = 0; i < Cols; i++) {
        deviations[i] = sqrt(deviations[i] / Rows);
    }

    // standardizing each value in given matrix with the respective mean and standard deviation
    for (int i  = 0; i < Rows; i++) {
        for (int j = 0; j < Cols; j++) {
            pMat[i][j] = (pMat[i][j] - means[j]) / deviations[j];
        }
    }
}

Matrix Matrix::dropColumn(int removedColumn) {
    if (removedColumn < 0 || removedColumn >= Cols) {
        throw invalid_argument("column to be removed is out of range (dropColumn)");
    }

    Matrix newMat = Matrix(Rows, Cols - 1);
    int columnIndex = 0;
    for (int i  = 0; i < Rows; i++) {
        columnIndex = 0;
        for (int j = 0; j < Cols; j++) {
            if (j == removedColumn) {
                continue;
            }
            newMat.pMat[i][columnIndex] = pMat[i][j];
            columnIndex++;
        }
    }

    return newMat;
}

void Matrix::shuffleRows() {
    Matrix newMat = Matrix(Rows, Cols); 

    // filling a vector with all valid indices
    vector<int> validIndices;
    for (int i = 0; i < Rows; i++) {
        validIndices.push_back(i);
    }
    // randomly ordering the indices within the vector
    vector<int> indexSet;
    int randomIndex;
    for (int i = 0; i < Rows; i++) {
        randomIndex = validIndices.size() == 1 ? 0 : (int) (((rand() % 100) / 100.0) * validIndices.size());
        indexSet.push_back(validIndices[randomIndex]);
        validIndices.erase(validIndices.begin() + randomIndex);
    }

    // filling the newMat in the randomly chosen order
    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j < Cols; j++) {
            newMat.setMatrixValue(i, j, pMat[indexSet[i]][j]);
        }
    }

    setMatrix(newMat);
}

void Matrix::replaceValueInColumn(int col, double oldValue, double newValue) {
    if (col < 0 || col >= Cols) {
        throw invalid_argument("column to be removed is out of range (dropColumn)");
    }

    for (int i = 0; i < Rows; i++) {
        pMat[i][col] = pMat[i][col] == oldValue ? newValue : pMat[i][col];
    }
}

Matrix Matrix::addRowWeight(Matrix weight) {

    if (weight.getRows() != 1 || weight.getCols() != Cols) {
        throw invalid_argument("invalid dimensions for successful weighting (addRowWeight)");
    }

    Matrix newMat = Matrix(Rows, Cols);

    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j < Cols; j++) {
            newMat.setMatrixValue(i, j, pMat[i][j] + weight.getMatrixValue(0, j));
        }
    }

    return newMat;
}

// PRIVATE METHODS -------------------------------------------------

double Matrix::boxMuller() {
    double u1 = (rand() % 100) / 100.0;
    double u2 = (rand() % 100) / 100.0;
    return sqrt(-2 * log(Global::ETAF(u1))) * cos(2 * Global::PI * u2);
}

void Matrix::allocateMemory() {
    pMat = new double*[Rows];
    for (int i = 0; i < Rows; ++i) {
        pMat[i] = new double[Cols];
    }
}

