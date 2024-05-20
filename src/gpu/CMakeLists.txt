set(SOURCE_FILES allocation_gpu.cu ../../include/allocation_gpu.cuh
        assert.cuh
        verify.cu)

target_sources(nnlib PRIVATE ${SOURCE_FILES})
