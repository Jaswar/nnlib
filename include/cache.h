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
#include <stack>
#include <unordered_map>
#include "allocation_gpu.cuh"

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
    std::unordered_map<size_t, std::stack<void*>> hostCache;
    std::unordered_map<size_t, std::stack<void*>> deviceCache;

    Cache() {
        hostCache = std::unordered_map<size_t, std::stack<void*>>();
        deviceCache = std::unordered_map<size_t, std::stack<void*>>();
    }

public:
    static Cache& getInstance() {
        static Cache instance;
        return instance;
    }

    template <typename T>
    T* get(size_t size, DataLocation location) {
        if (size == 0) {
            return nullptr;
        }

        size = size * sizeof(T);
        if (location == HOST) {
            auto it = hostCache.find(size);
            if (it != hostCache.end() && !it->second.empty()) {
                T* ptr = static_cast<T*>(it->second.top());
                it->second.pop();
                return ptr;
            } else {
                return allocate1DArray<T>(size);
            }
        } else {
            auto it = deviceCache.find(size);
            if (it != deviceCache.end() && !it->second.empty()) {
                T* ptr = static_cast<T*>(it->second.top());
                it->second.pop();
                return ptr;
            } else {
                return allocate1DArrayDevice<T>(size);
            }
        }
    }

    template<typename T>
    void put(size_t size, T* ptr, DataLocation location) {
        if (size == 0) {
            return;
        }

        size = size * sizeof(T);
        if (location == HOST) {
            auto it = hostCache.find(size);
            if (it == hostCache.end()) {
                hostCache.insert(std::pair<size_t, std::stack<void*>>(size, std::stack<void*>()));
            }
            it = hostCache.find(size);
            it->second.push(static_cast<void*>(ptr));
        } else {
            auto it = deviceCache.find(size);
            if (it == deviceCache.end()) {
                deviceCache.insert(std::pair<size_t, std::stack<void*>>(size, std::stack<void*>()));
            }
            it = deviceCache.find(size);
            it->second.push(static_cast<void*>(ptr));
        }
    }

    ~Cache() {
        for (auto& pair : hostCache) {
            while (!pair.second.empty()) {
                void* ptr = pair.second.top();
                free(ptr);
                pair.second.pop();
            }
        }
        // cuda memory cannot be deallocated because the cuda context is already destroyed
        // hopefully cuda will deallocate the memory itself
    }

    Cache(Cache const&) = delete;
    void operator=(Cache const&) = delete;
};


#endif //NNLIB_CACHE_H
