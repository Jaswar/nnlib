set(SOURCE_FILES activation.h allocation.h layer.h network.h onehot_encode.h read.h verify.cuh tensor.h session.cuh)

target_sources(nnlib PRIVATE ${SOURCE_FILES})