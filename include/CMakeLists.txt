set(INCLUDE_FILES
        metric.h
        activation.h
        allocation.h
        layer.h
        network.h
        onehot_encode.h
        read.h
        verify.cuh
        tensor.h
        session.cuh
        loss.h
        cache.h
        functions.h
        tuple_utils.h

        PARENT_SCOPE)

target_sources(nnlib PRIVATE ${INCLUDE_FILES})