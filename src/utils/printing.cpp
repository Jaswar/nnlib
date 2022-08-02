//
// Created by Jan Warchocki on 05/04/2022.
//

#include "printing.h"
#include <iostream>
#include <utility>

std::string constructFinishedProgressBar() {
    std::string result = "[";
    for (int step = 0; step < 20; step++) {
        result += "=";
    }
    result += "]";
    return result;
}

std::string constructProgressBar(size_t currentStep, size_t maxSteps) {
    if (maxSteps == 0) {
        return constructFinishedProgressBar();
    }
    if (currentStep >= maxSteps) {
        return constructFinishedProgressBar();
    }

    double currentProgress = static_cast<double>(currentStep) / static_cast<double>(maxSteps);
    double increment = 1.0 / 20;

    std::string result = "[";
    for (int step = 0; step < 20; step++) {
        double percentage = step * increment;
        double nextPercentage = (step + 1) * increment;

        if (currentProgress >= percentage) {
            if (currentProgress <= nextPercentage) {
                result += ">";
            } else {
                result += "=";
            }
        } else {
            result += "-";
        }
    }
    result += "]";

    return result;
}

std::string constructPercentage(size_t currentStep, size_t maxSteps) {
    if (maxSteps == 0) {
        return "[0/0 (100%)]";
    }
    if (currentStep >= maxSteps) {
        return "[" + std::to_string(maxSteps) + "/" + std::to_string(maxSteps) + " (100%)]";
    }

    int percentage = static_cast<int>(static_cast<double>(currentStep) / static_cast<double>(maxSteps) * 100);

    return "[" + std::to_string(currentStep) + "/" + std::to_string(maxSteps)
                + " (" + std::to_string(percentage) + "%)]";
}

std::string constructTime(size_t milliseconds) {
    size_t mil = milliseconds % 1000;
    size_t seconds = (milliseconds / 1000) % 60;
    size_t minutes = (milliseconds / (1000LL * 60)) % 60;
    size_t hours = (milliseconds / (1000LL * 60 * 60));

    return "(" + std::to_string(hours) + "h " + std::to_string(minutes) + "m "
            + std::to_string(seconds) + "s " + std::to_string(mil) + "ms)";
}
