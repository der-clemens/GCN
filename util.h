//
// Created by Clemens Hartmann on 24/08/2023.
//

#ifndef GCN_UTIL_H
#define GCN_UTIL_H

#include "blis.h"
#include "Graph.h"
#include "Matrix.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>

using std::string;
using std::ifstream;
using std::ostringstream;
using std::istringstream;

string readFileIntoString(const string& path) {
    auto ss = ostringstream{};
    ifstream input_file(path);
    if (!input_file.is_open()) {
        std::cerr << "Could not open the file - '"
                  << path << "'\n";
        exit(EXIT_FAILURE);
    }
    ss << input_file.rdbuf();
    return ss.str();
}

Graph readGraph(const string& path) {
    string file = readFileIntoString(path);
    istringstream sstream(file);
    string line;

    std::getline(sstream, line);

    std::getline(sstream, line);
    int vertexCount = std::stoi(line);
    std::getline(sstream, line);
    int edgeCount = std::stoi(line);

    auto* vertices = new size_t[vertexCount+1] {0};
    vertices[vertexCount] = edgeCount;
    auto edges = new size_t[edgeCount] {0};

    for(int i = 0; i < vertexCount; i++) {
        std::getline(sstream, line);
        int num = std::stoi(line);
        vertices[i] = num;
    }
    for(int i = 0; i < edgeCount; i++) {
        std::getline(sstream, line);
        int num = std::stoi(line);
        edges[i] = num;
    }

    auto GA = Graph(vertexCount, edgeCount);
    GA.index = vertices;
    GA.edges = edges;
    return GA;
}

Matrix readCSV(const string& path, char delimeter) {
    string file_contents = readFileIntoString(path);

    auto rows = std::count(file_contents.begin(), file_contents.end(), '\n') + 1;
    auto seperatorCount = std::count(file_contents.begin(), file_contents.end(), delimeter);
    auto columns = (seperatorCount+rows)/rows;

    auto result = Matrix(rows, columns);

    istringstream sstream(file_contents);
    string record;
    size_t i = 0;
    while (std::getline(sstream, record)) {
        size_t j = 0;
        istringstream line(record);
        while (std::getline(line, record, delimeter)) {
            int num = std::stoi(record);
            result.set(i,j, num);
            j++;
        }
        i++;
    }
    return result;
}

obj_t* readCSVtoVector(const string& path, char delimeter) {
    string file_contents = readFileIntoString(path);

    auto rows = std::count(file_contents.begin(), file_contents.end(), '\n') + 1;
    auto seperatorCount = std::count(file_contents.begin(), file_contents.end(), delimeter);
    auto columns = (seperatorCount+rows)/rows;

    auto result = new obj_t;
    bli_obj_create(BLIS_FLOAT, rows, 1, 0, 0, result);

    istringstream sstream(file_contents);
    string record;
    size_t i = 0;
    while (std::getline(sstream, record)) {
        size_t j = 0;
        istringstream line(record);
        while (std::getline(line, record, delimeter)) {
            int num = std::stoi(record);
            bli_setijm(num, 0, i, j, result);
            j++;
        }
        i++;
    }
    return result;
}

#endif //GCN_UTIL_H
