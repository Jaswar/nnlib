/**
 * @file function.h
 * @brief
 *
 * @author Jan Warchocki
 * @date 03 May 2024
 *
 */

#ifndef NNLIB_FUNCTION_H
#define NNLIB_FUNCTION_H

#include <utility>

#include "tensor.h"

class Function {
public:
    std::vector<sTensor> parents;

    Function() = default;

    sTensor forward(const std::vector<sTensor>& args) {
        parents = args;
        sTensor result = forwardFn(args);
        for (auto& parent : parents) {
            if (parent->requiresGrad) {
                result->requiresGrad = true;
                result->gradFunction = this;
                break;
            }
        }
        return result;
    }

    std::vector<sTensor> backward(sTensor grad) {
        return backwardFn(std::move(grad));
    }

    virtual sTensor forwardFn(const std::vector<sTensor>& args) = 0;

    virtual std::vector<sTensor> backwardFn(sTensor grad) = 0;
};

class Matmul : public Function {
    sTensor cacheA;
    sTensor cacheB;
public:
    sTensor forwardFn(const std::vector<sTensor>& args) override;

    std::vector<sTensor> backwardFn(sTensor grad) override;
};


#endif //NNLIB_FUNCTION_H
