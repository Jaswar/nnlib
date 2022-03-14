//
// Created by Jan Warchocki on 14/03/2022.
//

#ifndef NNLIB_DIFFERENT_DATA_LOCATION_EXCEPTION_H
#define NNLIB_DIFFERENT_DATA_LOCATION_EXCEPTION_H


#include <exception>

class DifferentDataLocationException : public std::exception {
    const char * what() const throw ();
};


#endif //NNLIB_DIFFERENT_DATA_LOCATION_EXCEPTION_H
