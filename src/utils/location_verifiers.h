//
// Created by Jan Warchocki on 28/05/2022.
//

#ifndef NNLIB_LOCATION_VERIFIERS_H
#define NNLIB_LOCATION_VERIFIERS_H

#include <vector.h>

bool allLocationsAreSame(std::initializer_list<DataLocation> locations);

bool allLocationsAreHost(std::initializer_list<DataLocation> locations);

bool allLocationsAreDevice(std::initializer_list<DataLocation> locations);

#endif //NNLIB_LOCATION_VERIFIERS_H
