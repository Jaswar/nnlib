/**
 * @file loss.cpp
 * @brief Source file defining the methods of the Loss class.
 * @author Jan Warchocki
 * @date 25 December 2022
 */

#include <loss.h>

Loss::Loss() : Metric() {
}

float Loss::calculateMetric(const sTensor& targets, const sTensor& predictions) {
    sTensor loss = calculateLoss(targets, predictions)->copy();
    loss->move(HOST);
    float l = loss->data[0];
    numSamples += targets->shape[0];
    currentTotalMetric += l;

    return currentTotalMetric / static_cast<float>(numSamples);
}
