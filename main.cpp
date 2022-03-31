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

    Matrix X = readCSV("C:/Users/janwa/CLionProjects/nnlib/data/features.txt");
    std::cout << X << std::endl;

    X.moveToDevice();

    const Vector& yv = convertToVector(readCSV("C:/Users/janwa/CLionProjects/nnlib/data/targets.txt"));
    Matrix y = oneHotEncode(yv);
    std::cout << y << std::endl;

    y.moveToDevice();

    Network network = Network(10);
    network.add(14);
    network.add(7, "sigmoid");

    network.train(X, y, 300, 32, 0.01);

    return 0;
}
