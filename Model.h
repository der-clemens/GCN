//
// Created by Clemens Hartmann on 22/09/2023.
//

#ifndef GCN_MODEL_H
#define GCN_MODEL_H

#include <chrono>
#include <vector>
#include <iostream>
#include <iomanip>
#include "GraphMatrix.h"
#include "Layer.h"

using Clock = std::chrono::steady_clock;
using Duration = std::chrono::duration<double>;

#define TIC(id) auto id  = Clock::now()
#define TOC(id,msg) do {\
	Duration diff = Clock::now() - id; \
	std::cout << msg << ": " << diff.count() << "\n"; \
} while(0)

class Model {
public:
    GraphMatrix A;
    const int layerCount;
    std::vector<Layer> layers;

    Model(const Graph& _graph, size_t* _filterCounts, int _layerCount);

    void forwardPass(const Matrix& x);
    void fit(Matrix& x, const Matrix& y, const Matrix& train, const size_t epochs, const float learningRate, const float weightDecay);
    Matrix classify(const Matrix& out);
    float accuracy(const Matrix& y, const Matrix& y_predicted);
    float trainAccuracy(const Matrix& train, const Matrix& y, const Matrix& y_predicted);

};


#endif //GCN_MODEL_H
