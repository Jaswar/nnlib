set(SOURCE_FILES allocation.cpp allocation.h
        onehot_encode.cpp onehot_encode.h
        read.cpp read.h
        printing.cpp printing.h)

target_sources(nnlib PRIVATE ${SOURCE_FILES})
