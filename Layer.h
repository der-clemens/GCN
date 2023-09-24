//
// Created by Clemens Hartmann on 21/09/2023.
//

#ifndef GCN_LAYER_H
#define GCN_LAYER_H


#include "GraphMatrix.h"
#include "Matrix.h"

class Layer {
public:
    //Â Matrix
    GraphMatrix A;

    //Layer Weights
    Matrix w;
    //Activation Function
    void (*activation)(Matrix&);
    Matrix out;

    Layer(GraphMatrix& _A, size_t _inFeatures, size_t _outFeatures, void (*_activation)(Matrix&));

    void forward(const Matrix& x);
    Matrix* backpropagate(Matrix* dz);

    void updateWeights(const Matrix& dz, Matrix& prevIn, float learningRate, float scaledWeightDecay);
};


#endif //GCN_LAYER_H
