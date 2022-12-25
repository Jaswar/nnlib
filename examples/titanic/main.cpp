/**
 * @file main.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 25 December 2022
 */

#include <iostream>
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

    Tensor dataset = readCSV(argv[1], ",", 4);
    Tensor X = Tensor(dataset.shape[0], dataset.shape[1] - 1);
    Tensor y = Tensor(dataset.shape[0], 1);

    for (int i = 0; i < dataset.shape[0]; i++) {
        y.data[i] = dataset.data[i * dataset.shape[1] + dataset.shape[1] - 1];
        for (int j = 0; j < dataset.shape[1] - 1; j++) {
            X.data[i * X.shape[1] + j] = dataset.data[i * dataset.shape[1] + j] / 255;
        }
    }

    std::cout << y << std::endl;
    std::cout << X << std::endl;

    Network network = Network(X.shape[1]);
    network.add(64);
    network.add(y.shape[1], "sigmoid");

    network.train(X, y, 25, 10, 0.05, new BinaryCrossEntropy());

    return 0;
}