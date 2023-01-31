#include<vector>

#ifndef MATRIX_H
#define MATRIX_H

using namespace std;

class Matrix {

    int Rows;
    int Cols;
    double** pMat;

public:

    // CONSTRUCTORS and DESTRUCTORS --------------------

    Matrix(const int rows, const int cols);

    Matrix(const int rows, const int cols, const vector<vector<double>> mat);

    Matrix();

    ~Matrix();

    Matrix(const Matrix&);
    
    Matrix& operator=(const Matrix&);

    // GETTER METHODS --------------------------

    int getRows() const;
    
    int getCols() const;

    double getMatrixValue(int i, int j) const;

    Matrix getMatrixRow(int rowIndex) const;

    Matrix getMatrixColumn(int colIndex) const;

    // SETTER METHODS ---------------------------

    void setMatrix(Matrix mat);

    void setMatrixValue(int i, int j, double value);

    void setMatrixRow(int rowIndex, Matrix mat);

    // OTHER METHODS -----------------------------

    // initializes random 2D matrix following a uniform normal distribution (using the Box-Muller method)
    void randomizeMatrix();

    // returns a Matrix with the given function applied to each value in the Matrix on which it is called
    Matrix applyMatrixOperation(double (*func)(double));

    Matrix applyMatrixOperationsInOrder(double (*func1)(double), double (*func2)(double));

    Matrix scaleBy(double scaleValue);

    // returns dot product / matrix multiplication of two Matrices
    Matrix dot(Matrix mat);

    Matrix transpose();

    void standardizeColumns();

    Matrix dropColumn(int removedColumn);

    void shuffleRows();

    void replaceValueInColumn(int col, double value, double newValue);

    Matrix addRowWeight(Matrix weight);

private:

    double boxMuller();

    void allocateMemory();
};

#endif