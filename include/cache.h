/**
 * @file cache.h
 * @brief
 *
 * @author Jan Warchocki
 * @date 23 April 2024
 *
 */

#ifndef NNLIB_CACHE_H
#define NNLIB_CACHE_H

#include <cstdlib>
#include <unordered_map>

/**
 * @brief Enumerate to specify where data is located.
 *
 * Can be either HOST or DEVICE. In case it is set to HOST, the data is stored in RAM and is processed by the CPU.
 * In case it is set to DEVICE, the data is in VRAM and processed by the GPU. The latter is only possible if CUDA
 * is installed and there is a CUDA enabled GPU on the system.
 */
enum DataLocation { HOST, DEVICE };


class Cache {
private:
    std::unordered_multimap<size_t, float*> hostCache;
    std::unordered_multimap<size_t, float*> deviceCache;

    Cache();

public:
    static Cache& getInstance() {
        static Cache instance;
        return instance;
    }

    float* get(size_t size, DataLocation location);

    void put(size_t size, float* ptr, DataLocation location);

    ~Cache();

    Cache(Cache const&) = delete;
    void operator=(Cache const&) = delete;
};


#endif //NNLIB_CACHE_H
