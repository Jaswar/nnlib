set(SOURCE_FILES allocation.cpp
        onehot_encode.cpp
        read.cpp
        printing.cpp printing.h
        location_verifiers.cpp
        session.cu)

target_sources(nnlib PRIVATE ${SOURCE_FILES})
