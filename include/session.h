/**
 * @file session.h
 * @brief 
 * @author Jan Warchocki
 * @date 29 August 2022
 */

#ifndef NNLIB_SESSION_H
#define NNLIB_SESSION_H

#include <thread>

class Session {
public:
    unsigned int threadsPerBlock;
    unsigned int numCores;

    Session();
};

#endif //NNLIB_SESSION_H
