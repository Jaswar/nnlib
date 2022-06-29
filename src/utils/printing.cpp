//
// Created by Jan Warchocki on 05/04/2022.
//

#include <utility>
#include <iostream>
#include "printing.h"

void showProgressBarFinished(int maxSteps) {
    std::cout << "\r[";
    for (int j = 0; j < 20; j++) {
        std::cout << "=";
    }
    std::cout << "] (" << maxSteps << "/" << maxSteps << ")" << std::flush;
}

void showProgressBar(int currentStep, int maxSteps) {
    if (currentStep % (maxSteps / 20) == 0) {
        std::cout << "\r[";
        for (int j = 0; j < 20; j++) {
            if (j < currentStep / (maxSteps / 20)) {
                std::cout << "=";
            } else if (j == currentStep / (maxSteps / 20)) {
                std::cout << ">";
            } else {
                std::cout << "-";
            }
        }
        std::cout << "] (" << currentStep << "/" << maxSteps << ")" << std::flush;
    } else if (currentStep >= maxSteps) {
        showProgressBarFinished(maxSteps);
    }
}
