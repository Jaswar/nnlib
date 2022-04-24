set(SOURCE_FILES activation.cu activation.cuh
        backpropagation.cu backpropagation.cuh
        layer.cpp ../../include/layer.h
        network.cpp ../../include/network.h)

target_sources(nnlib PRIVATE ${SOURCE_FILES})
