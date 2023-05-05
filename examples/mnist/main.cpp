/**
 * @file main.cpp
 * @brief Source file containing an example usage of the library for digit recognition.
 * @author Jan Warchocki
 * @date 25 April 2022
 */

#include <iostream>
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
 * @include mnist/main.cpp
 *
 * The project can be built with the following CMake script. This script requires `CMAKE_PREFIX_PATH` to be set to
 * the install directory of `nnlib`.
 * @include mnist/CMakeLists.txt
 */
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Dataset file path was not specified." << std::endl;
        return 1;
    }

    showCudaInfo();

    Tensor dataset = readCSV(argv[1], ",", 4);
    Tensor X = Tensor(dataset.shape[0], dataset.shape[1] - 1);
    Tensor yv = Tensor(dataset.shape[0]);

    for (int i = 0; i < dataset.shape[0]; i++) {
        yv.data[i] = dataset.data[i * dataset.shape[1] + 0];
        for (int j = 1; j < dataset.shape[1]; j++) {
            X.data[i * X.shape[1] + j - 1] = dataset.data[i * dataset.shape[1] + j] / 255;
        }
    }

    Tensor y = oneHotEncode(yv);

    std::cout << y << std::endl;

    Network network = Network(X.shape[1], true);
    network.add(64);
    network.add(y.shape[1], "sigmoid");

    std::vector<Metric*> metrics = {new CategoricalAccuracy(), new MeanSquaredError()};
    network.train(X, y, 25, 10, 0.01, new CategoricalCrossEntropy(), metrics);

    return 0;
}
