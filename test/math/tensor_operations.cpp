//
// Created by Jan Warchocki on 26/08/2022.
//

#include <gtest/gtest.h>
#include <vector.h>
#include <verify.cuh>
#include "../utils.h"
#include "../assertions.h"
#include <tensor.h>

TEST(tensor, test) {
    Tensor t1 = Tensor(4, 3, 3, 3);
    t1.host[9] = 5;

    std::cout << t1 << std::endl;

}
