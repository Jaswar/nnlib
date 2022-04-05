//
// Created by Jan Warchocki on 06/03/2022.
//

#ifndef NNLIB_READ_H
#define NNLIB_READ_H

#include <string>
#include "../math/matrix.h"
#include <vector>

bool fileExists(const std::string& filepath);

std::vector<std::string> readFile(const std::string& filepath);

Matrix readCSV(const std::string& filepath, const std::string& delim = ",", int numThreads = 1);


#endif //NNLIB_READ_H
