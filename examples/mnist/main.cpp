#include <iostream>
#include <nnlib/matrix.h>
#include <nnlib/network.h>
#include <nnlib/read.h>
#include <nnlib/verify.cuh>
#include <nnlib/onehot_encode.h>
#include <chrono>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Dataset file path was not specified." << std::endl;
        return 1;
    }

    showCudaInfo();

    Matrix dataset = readCSV(argv[1], ",", 4);
    Matrix X = Matrix(dataset.n, dataset.m - 1, HOST);
    Vector yv = Vector(dataset.n, HOST);

    for (int i = 0; i < dataset.n; i++) {
        yv[i] = dataset(i, 0);
        for (int j = 1; j < dataset.m; j++) {
            X(i, j - 1) = dataset(i, j) / 255;
        }
    }

    Matrix y = oneHotEncode(yv);

    std::cout << y << std::endl;

    if (isCudaAvailable()) {
        X.moveToDevice();
        y.moveToDevice();
    }

    Network network = Network(X.m);
    network.add(64);
    network.add(y.m, "sigmoid");

    network.train(X, y, 25, 10, 0.01);

    return 0;
}
