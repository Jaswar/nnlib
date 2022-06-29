//
// Created by Jan Warchocki on 06/03/2022.
//

#include <utility>
#include <fstream>
#include "read.h"
#include "printing.h"
#include <thread>
#include <cmath>

std::vector<std::string> split(const std::string& str, const std::string& delim) {
    size_t start = 0, end = 0;

    std::vector<std::string> result;
    while ((end = str.find(delim, start)) != std::string::npos) {
        result.push_back(str.substr(start, end - start));
        start = end + delim.length();
    }
    result.push_back(str.substr(start));

    return result;
}

bool fileExists(const std::string& filepath) {
    std::ifstream file(filepath);
    bool exists = file.is_open();
    file.close();
    return exists;
}

std::vector<std::string> readFile(const std::string& filepath) {
    std::vector<std::string> lines;
    std::ifstream file;

    if (fileExists(filepath)) {
        file = std::ifstream(filepath);
    } else {
        file = std::ifstream("../" + filepath);
    }

    std::string line;
    if (file.is_open()) {
        while (std::getline(file, line)) {
            lines.push_back(line);
        }
        file.close();
    }

    return lines;
}

void threadCSVJob(const std::vector<std::string>& lines, const std::string& delim, Matrix& result, int id, int numThreads) {
    int size = static_cast<int>(lines.size());
    int numIterations = std::ceil(size / static_cast<double>(numThreads));

    for (int i = 0; i < numIterations; i++) {
        if (id == 0) {
            showProgressBar(i * numThreads, size);
        }
        int index = id + numThreads * i;
        if (index >= lines.size()) {
            return;
        }

        const std::string& line = lines.at(index);
        size_t start = 0, end;
        int j = 0;
        while ((end = line.find(delim, start)) != std::string::npos) {
            result(index, j++) = static_cast<DTYPE>(std::stod(line.substr(start, end - start)));
            start = end + delim.length();
        }
        result(index, j) = static_cast<DTYPE>(std::stod(line.substr(start)));
    }

    if (id == 0) {
        showProgressBar(size, size);
    }
}

Matrix readCSV(const std::string& filepath, const std::string& delim, int numThreads) {
    std::cout << "Reading CSV file " << filepath << std::endl;

    auto lines = readFile(filepath);

    auto n = lines.size();
    auto m = n > 0 ? split(lines.front(), delim).size() : 1;
    Matrix result = Matrix(n, m);

    std::vector<std::thread> threads;
    for (int id = 0; id < numThreads; id++) {
        std::thread thread(threadCSVJob, std::ref(lines), std::ref(delim), std::ref(result), id, numThreads);
        threads.push_back(std::move(thread));
    }

    for (auto& thread : threads) {
        thread.join();
    }

    std::cout << std::endl;

    return result;
}
