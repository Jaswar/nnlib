//
// Created by Jan Warchocki on 27/06/2022.
//

#ifndef NNLIB_UTILS_H
#define NNLIB_UTILS_H

#include <matrix.h>
#include <vector.h>

Vector constructVector(std::initializer_list<DTYPE> vectorDefinition, DataLocation location = HOST);
Matrix constructMatrix(std::initializer_list<std::initializer_list<DTYPE>> matrixDefinition,
                       DataLocation location = HOST);

#endif //NNLIB_UTILS_H
