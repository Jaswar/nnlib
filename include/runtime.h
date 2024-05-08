/**
 * @file runtime.h
 * @brief
 *
 * @author Jan Warchocki
 * @date 08 May 2024
 *
 */

#ifndef NNLIB_RUNTIME_H
#define NNLIB_RUNTIME_H

class Runtime {
    Runtime();

public:
    bool useGradient;

    static Runtime& getInstance() {
        static Runtime instance;
        return instance;
    }

    void disableGradient();

    void enableGradient();

    ~Runtime() = default;

    Runtime(Runtime const&) = delete;
    void operator=(Runtime const&) = delete;
};

#endif //NNLIB_RUNTIME_H
