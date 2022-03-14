//
// Created by Jan Warchocki on 06/03/2022.
//

#include <utility>
#include <fstream>
#include "read.h"

std::vector<std::string> split(const std::string& str, const std::string& delim) {
    size_t start = 0;

    std::vector<std::string> result;
    while (true) {
        size_t end = str.find(delim, start);
        if (end == std::string::npos) {
            break;
        }

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

Matrix readCSV(const std::string& filepath, const std::string& delim) {
    auto lines = readFile(filepath);

    auto n = static_cast<int>(lines.size());
    auto m = n > 0 ? static_cast<int>(split(lines.front(), delim).size()) : 1;
    Matrix result = Matrix(n, m);

    // Read every line
    for (auto it = lines.begin(); it < lines.end(); it++) {
        size_t i = it - lines.begin();
        auto values = split(*it, delim);

        // Split every line by the delimiter and iterate over the values
        DTYPE* row = allocate1DArray(m);
        for (auto it2 = values.begin(); it2 < values.end(); it2++) {
            size_t j = it2 - values.begin();

            row[j] = static_cast<DTYPE>(std::stod(*it2));
        }

        for (int j = 0; j < m; j++) {
            result(i, j) = row[j];
        }
    }

    return result;
}
