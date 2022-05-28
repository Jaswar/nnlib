//
// Created by Jan Warchocki on 28/05/2022.
//

#ifndef NNLIB_UNEXPECTED_CUDA_CALL_EXCEPTION_H
#define NNLIB_UNEXPECTED_CUDA_CALL_EXCEPTION_H

#include <exception>

class UnexpectedCUDACallException : public std::exception {
    const char * what() const throw ();
};


#endif //NNLIB_UNEXPECTED_CUDA_CALL_EXCEPTION_H
