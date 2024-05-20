set(INCLUDE_FILES
        metric.h
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
        runtime.h
        exceptions.h
        tensor_operations_on_device.cuh
        tensor_operations_on_host.h
        location_verifiers.h
        allocation_gpu.cuh

        PARENT_SCOPE)

target_sources(nnlib PRIVATE ${INCLUDE_FILES})