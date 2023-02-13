/**
 * @file main.cpp
 * @brief Source file containing an example usage of the library for the Titanic dataset.
 * @author Jan Warchocki
 * @date 25 December 2022
 */

#include <iostream>
#include <nnlib/network.h>
#include <nnlib/read.h>
#include <nnlib/verify.cuh>
#include <nnlib/onehot_encode.h>
#include <chrono>

/**
 * @example Titanic
 * Training a neural network on the Titanic dataset.
 *
 * The dataset can be downloaded from https://www.kaggle.com/c/titanic. Since `nnlib` does not support feature
 * engineering, the dataset needs to be prepared using the following Python script:
 * @include titanic/prepare.py
 *
 * The Python script expects one parameter, which is the full path to the Titanic dataset. The dataset prepared for
 * `nnlib` will then be generated in the `./out` directory.
 *
 * The `main.cpp` file expects one argument which is the absolute path to the file prepared by the Python script.
 * @include titanic/main.cpp
 *
 * The project can be built with the following CMake script. This script requires `CMAKE_PREFIX_PATH` to be set to
 * the install directory of `nnlib`.
 * @include titanic/CMakeLists.txt
 */
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Dataset file path was not specified." << std::endl;
        return 1;
    }

    showCudaInfo();

    Tensor dataset = readCSV(argv[1], ",", 4);
    Tensor X = Tensor(dataset.shape[0], dataset.shape[1] - 1);
    Tensor y = Tensor(dataset.shape[0], 1);

    for (int i = 0; i < dataset.shape[0]; i++) {
        y.data[i] = dataset.data[i * dataset.shape[1] + dataset.shape[1] - 1];
        for (int j = 0; j < dataset.shape[1] - 1; j++) {
            X.data[i * X.shape[1] + j] = dataset.data[i * dataset.shape[1] + j];
        }
    }

    std::cout << y << std::endl;

    Network network = Network(X.shape[1]);
    network.add(64);
    network.add(y.shape[1], "sigmoid");

    network.train(X, y, 100, 10,  0.01, new BinaryCrossEntropy());

    return 0;
}