/**
 * @file metric.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 13 February 2023
 */

#include <metric.h>

Metric::Metric() : numSamples(0), currentTotalMetric(0) {
}

void Metric::reset(){
    numSamples = 0;
    currentTotalMetric = 0;
}


