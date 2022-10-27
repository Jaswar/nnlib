/**
 * @file read.h
 * @brief Header file declaring methods that can be used when reading from files.
 * @author Jan Warchocki
 * @date 06 March 2022
 */

#ifndef NNLIB_READ_H
#define NNLIB_READ_H

#include "tensor.h"
#include <string>
#include <vector>

/**
 * @brief Check if a file exists with the provided filepath.
 *
 * @param filepath The filepath of the file to check.
 * @return True if the file exists, false otherwise.
 */
bool fileExists(const std::string& filepath);

/**
 * @brief Read file from a filepath.
 *
 * Will attempt to read file with the provided filepath or in the parent directory of the filepath.
 *
 * The method can only read text files and will not work on binary files.
 *
 * @param filepath The path to the file.
 * @return The lines read from the file.
 */
std::vector<std::string> readFile(const std::string& filepath);

/**
 * @brief Read a csv file from a path.
 *
 * The method will first read all lines from the file and then divide the lines over `numThreads` threads to
 * split them into values. The values are then written into a matrix that is then returned.
 *
 * @param filepath The path to the csv file.
 * @param delim The delimiter to use when splitting data.
 * @param numThreads The number of threads to use when splitting data.
 * @return Matrix of data that was read from the csv file.
 */
Tensor readCSV(const std::string& filepath, const std::string& delim = ",", int numThreads = 1);


#endif //NNLIB_READ_H
