/**
 * @file session.h
 * @brief Header file declaring the Session class.
 * @author Jan Warchocki
 * @date 29 August 2022
 */

#ifndef NNLIB_SESSION_H
#define NNLIB_SESSION_H

#include <thread>

/**
 * @brief Contains information about the current session.
 *
 * This includes information about the number of CPU cores and the number of threads per block in case a CUDA enabled
 * GPU is available. Each Tensor contains its own Session object.
 */
class Session {
    /**
     * @brief The number of threads per block of the GPU.
     *
     * If there is no GPU or no CUDA, this value is initialized to 0.
     */
public:
    unsigned int threadsPerBlock;

    /**
     * @brief The number of logical CPU cores.
     */
    unsigned int numCores;

    /**
     * @brief Constructor that initializes the both members of the class.
     */
    Session();
};

#endif //NNLIB_SESSION_H
