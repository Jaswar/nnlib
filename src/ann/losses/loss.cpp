
/**
 * @file loss.cpp
 * @brief 
 * @author Jan Warchocki
 * @date 25 December 2022
 */

#include <loss.h>

Loss::Loss() : numSamples(0), currentTotalLoss(0) {

}

void Loss::reset() {
    numSamples = 0;
    currentTotalLoss = 0;
}


