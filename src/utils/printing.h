//
// Created by Jan Warchocki on 05/04/2022.
//

#ifndef NNLIB_PRINTING_H
#define NNLIB_PRINTING_H

#include <string>

// Construct progress bar of the form [==============>-----]
std::string constructProgressBar(int currentStep, int maxSteps);
// Construct percentage in the form [25/100 (25%)]
std::string constructPercentage(int currentStep, int maxSteps);

#endif //NNLIB_PRINTING_H
