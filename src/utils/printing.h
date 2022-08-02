//
// Created by Jan Warchocki on 05/04/2022.
//

#ifndef NNLIB_PRINTING_H
#define NNLIB_PRINTING_H

#include <string>

// Construct progress bar of the form [==============>-----]
std::string constructProgressBar(size_t currentStep, size_t maxSteps);
// Construct percentage in the form [25/100 (25%)]
std::string constructPercentage(size_t currentStep, size_t maxSteps);
// Construct time spent info in the form (0h 2m 5s 127ms)
std::string constructTime(size_t milliseconds);

#endif //NNLIB_PRINTING_H
