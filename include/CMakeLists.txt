set(SOURCE_FILES activation.h allocation.h convert.h layer.h network.h onehot_encode.h read.h vector.h verify.cuh tensor.h)

target_sources(nnlib PRIVATE ${SOURCE_FILES})