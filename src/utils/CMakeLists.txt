set(SOURCE_FILES allocation.cpp
        onehot_encode.cpp
        read.cpp
        printing.cpp printing.h
        session.cu)

target_sources(nnlib PRIVATE ${SOURCE_FILES})
