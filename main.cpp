#include <iostream>
#include "math/matrix.h"
#include "ann/network.h"
#include "utils/read.h"
#include "gpu/verify.cuh"
#include "math/convert.h"
#include "utils/onehot_encode.h"

int main() {
    showCudaInfo();

    const Matrix& X = readCSV("data/features.txt");
    std::cout << X << std::endl;

    const Vector& yv = convertToVector(readCSV("data/targets.txt"));
    const Matrix& y = oneHotEncode(yv);
    std::cout << y << std::endl;

    Network network = Network(10);
    network.add(14);
    network.add(7);

    network.train(X, y, 25, 0.007);

    return 0;
}
