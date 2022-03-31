#include <iostream>
#include "math/matrix.h"
#include "ann/network.h"
#include "utils/read.h"
#include "gpu/verify.cuh"
#include "math/convert.h"
#include "utils/onehot_encode.h"
#include "ann/backpropagation.cuh"

int main() {
    showCudaInfo();

//    Matrix X = readCSV("C:/Users/janwa/CLionProjects/nnlib/data/features.txt");
//    std::cout << X << std::endl;
//
//    X.moveToDevice();
//
//    const Vector& yv = convertToVector(readCSV("C:/Users/janwa/CLionProjects/nnlib/data/targets.txt"));
//    Matrix y = oneHotEncode(yv);
//    std::cout << y << std::endl;

    Matrix dataset = readCSV("C:/Users/janwa/CLionProjects/nnlib/data/MNIST_test.txt");
    Matrix X = Matrix(dataset.n, dataset.m - 1, HOST);
    Vector yv = Vector(dataset.n, HOST);

    for (int i = 0; i < dataset.n; i++) {
        if (i % 1000 == 0) {
            std::cout << "Converting row " << i << std::endl;
        }
        yv[i] = dataset(i, 0);
        for (int j = 1; j < dataset.m; j++) {
            X(i, j - 1) = dataset(i, j) / 255;
        }
    }

    std::cout << yv << std::endl;

    Matrix y = oneHotEncode(yv);
    X.moveToDevice();
    y.moveToDevice();

    Network network = Network(X.m);
    network.add(64);
    network.add(y.m, "sigmoid");

    network.train(X, y, 25, 32, 0.01);

    return 0;
}
