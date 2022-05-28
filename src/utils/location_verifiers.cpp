//
// Created by Jan Warchocki on 28/05/2022.
//

#include "location_verifiers.h"

bool allLocationsAreSame(std::initializer_list<DataLocation> locations) {
    int countHost = 0;
    int countDevice = 0;
    for (auto location : locations) {
        if (location == HOST) {
            countHost++;
        } else {
            countDevice++;
        }
    }

    return countHost == locations.size() || countDevice == locations.size();
}

bool allLocationsAreHost(std::initializer_list<DataLocation> locations) {
    for (auto location : locations) {
        if (location == DEVICE) {
            return false;
        }
    }
    return true;
}

bool allLocationsAreDevice(std::initializer_list<DataLocation> locations) {
    for (auto location : locations) {
        if (location == HOST) {
            return false;
        }
    }
    return true;
}

