#include<vector>

#ifndef GLOBAL_H
#define GLOBAL_H

using namespace std;

class Global {

    public:

        static const double PI;
        static const double E;
        static const double ETA;

        // applies the ReLU function to a double value
        static double relu(double x);

        // applies the Sigmoid function to a double value
        static double sigmoid(double x);

        // applies the derivative the ReLU function to a double value
        static double dRelu(double x);

        // prevents a value of 0 from begin passed into the logarithm function, which results in infinity
        static double ETAF(double x);

        static double inverse(double x);

        // prints a Matrix  
        static void printMatrix(Matrix mat);    

        // returns the sum of values in a Matrix    
        static double sumMatrix(Matrix mat);

        static Matrix sumMatrix(Matrix mat, int axis);

        static Matrix combination(Matrix mat1, Matrix mat2, double (*func)(double, double));

        static Matrix addMatrices(Matrix mat1, Matrix mat2);

        static Matrix subtractMatrices(Matrix mat1, Matrix mat2);

        static Matrix multiplyMatrices(Matrix mat1, Matrix mat2);

        static Matrix divideMatrices(Matrix mat1, Matrix mat2);

        static double entropyLoss(Matrix y, Matrix yHat);

        static Matrix generateMatrix(string filePath);

        static vector<Matrix> splitData(Matrix X, Matrix y, double trainSize);

        static double getMeanSquaredError(Matrix Ytest, Matrix prediction);
};

#endif
