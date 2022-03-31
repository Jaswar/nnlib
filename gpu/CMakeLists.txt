set(SOURCE_FILES allocation_gpu.cu allocation_gpu.cuh
        assert.cuh
        verify.cu verify.cuh)

target_sources(nnlib PRIVATE ${SOURCE_FILES})
