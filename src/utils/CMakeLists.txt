set(SOURCE_FILES allocation.cpp ../../include/allocation.h
        onehot_encode.cpp ../../include/onehot_encode.h
        read.cpp ../../include/read.h
        printing.cpp printing.h)

target_sources(nnlib PRIVATE ${SOURCE_FILES})
