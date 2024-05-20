/**
 * @file binary_cross_entropy.cpp
 * @brief Source file defining the methods of the BinaryCrossEntropy class.
 * @author Jan Warchocki
 * @date 24 December 2022
 */

#include <exceptions.h>
#include <loss.h>

/**
 * @brief Check if the shape is valid for Binary Cross Entropy.
 *
 * The shape should be (n, 1) for both @p targets and @p predictions.
 *
 * @param targets The expected outputs of the network.
 * @param predictions The actual outputs of the network.
 */
template<typename T>
void checkValidShape(const Tensor<T>& targets, const Tensor<T>& predictions) {
    if (targets.shape.size() != 2 || targets.shape[1] != 1) {
        throw UnsupportedOperationException();
    }
    if (predictions.shape.size() != 2 || predictions.shape[1] != 1) {
        throw UnsupportedOperationException();
    }
}

std::string BinaryCrossEntropy::getShortName() const {
    return "binary_cross_entropy";
}

sfTensor BinaryCrossEntropy::calculateLoss(const sfTensor& targets, const sfTensor& predictions) {
    checkValidShape(*targets, *predictions);

    sfTensor totalLoss = std::make_shared<Tensor<float>>(targets->shape, targets->location);
    {
        sfTensor ones = std::make_shared<Tensor<float>>(targets->shape, targets->location);
        fill(1.0f, ones);
        sfTensor diffTargets = subtract(ones, targets);
        sfTensor diffPredictions = log(subtract(ones, predictions));
        totalLoss = hadamard(diffTargets, diffPredictions);
    }

    {
        sfTensor diffPredictions = log(predictions);
        totalLoss = add(hadamard(targets, diffPredictions), totalLoss);
    }

    totalLoss = multiply(sum(totalLoss), -1.0f);
    return totalLoss;
}
