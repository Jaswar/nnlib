/**
 * @file printing.h
 * @brief Header file declaring methods to construct useful string messages.
 * @author Jan Warchocki
 * @date 05 April 2022
 */

#ifndef NNLIB_PRINTING_H
#define NNLIB_PRINTING_H

#include <string>

/**
 * @brief Construct a string message representing a progress bar.
 *
 * The progress bar is displayed in the following format:
 * ```
 * [==============>-----]
 * ```
 *
 * @param currentStep The current progress to display.
 * @param maxSteps The maximum progress that can be achieved.
 * @return A string representing the progress bar.
 */
std::string constructProgressBar(size_t currentStep, size_t maxSteps);

/**
 * @brief Construct a string message about progress in percentage form.
 *
 * The info is displayed in the following format:
 * ```
 * [25/50 (50%)]
 * ```
 *
 * @param currentStep The current progress to display.
 * @param maxSteps The maximum progress that can be achieved.
 * @return A string representing the percentage.
 */
std::string constructPercentage(size_t currentStep, size_t maxSteps);

/**
 * @brief Construct a string message displaying current time spent.
 *
 * The time is displayed in the following format:
 * ```
 * (0h 2m 5s 127ms)
 * ```
 *
 * @param milliseconds The amount of time to display in milliseconds.
 * @return A string representing the time.
 */
std::string constructTime(size_t milliseconds);

#endif //NNLIB_PRINTING_H
