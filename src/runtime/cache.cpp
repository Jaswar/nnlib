/**
 * @file cache.cpp
 * @brief
 *
 * @author Jan Warchocki
 * @date 23 April 2024
 *
 */

#include "cache.h"
#include "../gpu/allocation_gpu.cuh"

Cache::Cache() {
    hostCache = std::unordered_multimap<size_t, float*>();
    deviceCache = std::unordered_multimap<size_t, float*>();
}

float* Cache::get(size_t size, DataLocation location) {
    if (size == 0) {
        return nullptr;
    }

    if (location == HOST) {
        auto it = hostCache.find(size);
        if (it != hostCache.end()) {
            float* ptr = it->second;
            hostCache.erase(it);
            return ptr;
        } else {
            return allocate1DArray(size);
        }
    } else {
        auto it = deviceCache.find(size);
        if (it != deviceCache.end()) {
            float* ptr = it->second;
            deviceCache.erase(it);
            return ptr;
        } else {
            return allocate1DArrayDevice(size);
        }
    }
}

void Cache::put(size_t size, float* ptr, DataLocation location) {
    if (size == 0) {
        return;
    }

    if (location == HOST) {
        hostCache.insert(std::pair<size_t, float*>(size, ptr));
    } else {
        deviceCache.insert(std::pair<size_t, float*>(size, ptr));
    }
}

Cache::~Cache() {
    for (auto& pair : hostCache) {
        free(pair.second);
    }
    for (auto& pair : deviceCache) {
        free1DArrayDevice(pair.second);
    }
}
