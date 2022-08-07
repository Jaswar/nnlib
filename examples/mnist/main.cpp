/**
 * @file main.cpp
 * @brief Source file containing an example usage of the library for digit recognition.
 * @author Jan Warchocki
 * @date 25 April 2022
 */

#include <iostream>
#include <nnlib/matrix.h>
#include <nnlib/network.h>
#include <nnlib/read.h>
#include <nnlib/verify.cuh>
#include <nnlib/onehot_encode.h>
#include <chrono>

/**
 * @example MNIST
 * Training a neural network to recognize hand-written digits from MNIST.
 *
 * Files taken from https://github.com/Jaswar/nnlib/tree/main/examples/mnist.
 *
 * The `main.cpp` file expects one argument which is the absolute path to the `MNIST_train.txt` file.
 * The file can be downloaded from https://github.com/halimb/MNIST-txt.
 * @include main.cpp
 *
 * The project can be built with the following CMake script. This script requires `CMAKE_PREFIX_PATH` to be set to
 * the install directory of nnlib.
 * @include CMakeLists.txt
 */
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
