/**
 * @file location_verifiers.h
 * @brief Header file declaring methods to verify multiple #DataLocation at once.
 * @author Jan Warchocki
 * @date 28 May 2022
 */

#ifndef NNLIB_LOCATION_VERIFIERS_H
#define NNLIB_LOCATION_VERIFIERS_H

#include <tensor.h>

/**
 * @brief Return true if all given locations are the same.
 *
 * @param locations The list of locations to verify.
 * @return True if all locations are the same, false otherwise.
 */
bool allLocationsAreSame(std::initializer_list<DataLocation> locations);

/**
 * @brief Return true if all given locations are HOST.
 *
 * @param locations The list of locations to verify.
 * @return True if all locations are HOST, false otherwise.
 */
bool allLocationsAreHost(std::initializer_list<DataLocation> locations);

/**
 * @brief Return true if all given locations are DEVICE.
 *
 * @param locations The list of locations to verify.
 * @return True if all locations are DEVICE, false otherwise.
 */
bool allLocationsAreDevice(std::initializer_list<DataLocation> locations);

#endif //NNLIB_LOCATION_VERIFIERS_H
