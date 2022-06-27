//
// Created by Jan Warchocki on 27/06/2022.
//

#ifndef NNLIB_UTILS_H
#define NNLIB_UTILS_H

#include <vector.h>
#include <matrix.h>

Vector constructVector(std::initializer_list<DTYPE> vectorDefinition);
Matrix constructMatrix(std::initializer_list<std::initializer_list<DTYPE>> matrixDefinition);

#endif //NNLIB_UTILS_H
