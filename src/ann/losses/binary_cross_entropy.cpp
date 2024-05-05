/**
 * @file binary_cross_entropy.cpp
 * @brief Source file defining the methods of the BinaryCrossEntropy class.
 * @author Jan Warchocki
 * @date 24 December 2022
 */

#include <exceptions/unsupported_operation_exception.h>
#include <loss.h>

/**
 * @brief Check if the shape is valid for Binary Cross Entropy.
 *
 * The shape should be (n, 1) for both @p targets and @p predictions.
 *
 * @param targets The expected outputs of the network.
 * @param predictions The actual outputs of the network.
 */
void checkValidShape(const Tensor& targets, const Tensor& predictions) {
    if (targets.shape.size() != 2 || targets.shape[1] != 1) {
        throw UnsupportedOperationException();
    }
    if (predictions.shape.size() != 2 || predictions.shape[1] != 1) {
        throw UnsupportedOperationException();
    }
}

float BinaryCrossEntropy::calculateLoss(const Tensor& targets, const Tensor& predictions) {
    checkValidShape(targets, predictions);

    Tensor totalLoss = Tensor(targets.shape);
    {
        Tensor ones = Tensor(targets.shape);
        fill(1, ones);
        Tensor diffTargets = subtract(ones, targets);
        Tensor diffPredictions = log(subtract(ones, predictions));
        totalLoss = hadamard(diffTargets, diffPredictions);
    }

    {
        Tensor diffPredictions = log(predictions);
        totalLoss = add(hadamard(targets, diffPredictions), totalLoss);
    }

    numSamples += targets.shape[0];
    currentTotalMetric += sum(totalLoss) * -1;

    return currentTotalMetric / static_cast<float>(numSamples);
}

Tensor BinaryCrossEntropy::calculateDerivatives(const Tensor& targets, const Tensor& predictions) {
    checkValidShape(targets, predictions);

    // Calculate the nominator
    Tensor result = subtract(predictions, targets);

    // Calculate the denominator
    Tensor ones = Tensor(targets.shape);
    fill(1, ones);
    Tensor denominator = subtract(ones, predictions);
    denominator = hadamard(predictions, denominator);

    // Calculate the fraction
    result = divide(result, denominator);
    return result;
}

std::string BinaryCrossEntropy::getShortName() const {
    return "binary_cross_entropy";
}

sTensor BinaryCrossEntropy::calculateLoss(const sTensor& targets, const sTensor& predictions) {
    checkValidShape(*targets, *predictions);

    sTensor totalLoss = std::make_shared<Tensor>(targets->shape);
    {
        sTensor ones = std::make_shared<Tensor>(targets->shape);
        ones->move(targets->location);
        fill(1.0f, ones);
        sTensor diffTargets = subtract(ones, targets);
        sTensor diffPredictions = log(subtract(ones, predictions));
        totalLoss = hadamard(diffTargets, diffPredictions);
    }

    {
        sTensor diffPredictions = log(predictions);
        totalLoss = add(hadamard(targets, diffPredictions), totalLoss);
    }

    totalLoss = multiply(sum(totalLoss), -1.0f);
    return totalLoss;
}
