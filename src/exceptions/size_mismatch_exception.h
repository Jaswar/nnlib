//
// Created by Jan Warchocki on 03/03/2022.
//

#ifndef NNLIB_SIZE_MISMATCH_EXCEPTION_H
#define NNLIB_SIZE_MISMATCH_EXCEPTION_H

#include <exception>

class SizeMismatchException : public std::exception {

    const char * what() const throw ();
};


#endif //NNLIB_SIZE_MISMATCH_EXCEPTION_H
