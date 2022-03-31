set(SOURCE_FILES activation.cu activation.cuh
        backpropagation.cu backpropagation.cuh
        layer.cpp layer.h
        network.cpp network.h)

target_sources(nnlib PRIVATE ${SOURCE_FILES})
