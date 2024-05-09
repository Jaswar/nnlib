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
#include <iostream>

Cache::Cache() {
    hostCache = std::unordered_map<size_t, std::stack<float*>>();
    deviceCache = std::unordered_map<size_t, std::stack<float*>>();
}

float* Cache::get(size_t size, DataLocation location) {
    if (size == 0) {
        return nullptr;
    }

    if (location == HOST) {
        auto it = hostCache.find(size);
        if (it != hostCache.end() && !it->second.empty()) {
            float* ptr = it->second.top();
            it->second.pop();
            return ptr;
        } else {
            return allocate1DArray(size);
        }
    } else {
        auto it = deviceCache.find(size);
        if (it != deviceCache.end() && !it->second.empty()) {
            float* ptr = it->second.top();
            it->second.pop();
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
        auto it = hostCache.find(size);
        if (it == hostCache.end()) {
            hostCache.insert(std::pair<size_t, std::stack<float*>>(size, std::stack<float*>()));
        }
        it = hostCache.find(size);
        it->second.push(ptr);
    } else {
        auto it = deviceCache.find(size);
        if (it == deviceCache.end()) {
            deviceCache.insert(std::pair<size_t, std::stack<float*>>(size, std::stack<float*>()));
        }
        it = deviceCache.find(size);
        it->second.push(ptr);
    }
}

Cache::~Cache() {
    for (auto& pair : hostCache) {
        while (!pair.second.empty()) {
            float* ptr = pair.second.top();
            free(ptr);
            pair.second.pop();
        }
    }
    // cuda memory cannot be deallocated because the cuda context is already destroyed
    // hopefully cuda will deallocate the memory itself
}
