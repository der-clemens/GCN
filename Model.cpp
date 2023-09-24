//
// Created by Clemens Hartmann on 22/09/2023.
//

#include "Model.h"

void relu(Matrix& a) noexcept {
    for(dim_t row_idx = 0; row_idx < a.rows; row_idx++) {
        for(dim_t col_idx = 0; col_idx < a.cols; col_idx++) {
            float ele = a.get(row_idx, col_idx);
            if(ele <= 0) {
                a.set(row_idx, col_idx, 0);
            }
        }
    }
}

void softmax(Matrix& in) {
    for (size_t v = 0; v < in.rows; v++) {
        float sum, max;
        // Calculate the max value for the node
        // Subtracting the max does not change the result of the function by prevents
        // the large values being produced after the eapplying the expondential
        max = 0;
        for (size_t i = 0; i < in.cols; i++) {
            float current = in.get(v, i);
            if(current > max)
                max = current;
        }

        // Perform the expontials and calculate sum
        sum = 0;
        for (size_t i = 0; i < in.cols; i++) {
            float current = in.get(v, i);
            float num = ::exp(current - max);
            sum += num;
            in.set(v, i, num);
        }

        // Divide each value by the sum
        for (size_t i = 0; i < in.cols; i++) {
            float current = in.get(v,i);
            in.set(v,i, current/sum);
        }
    }
}

Model::Model(const Graph &_graph, size_t *_filterCounts, int _layerCount) : A(GraphMatrix(_graph)), layerCount(_layerCount) {
    layers.reserve(layerCount);
    for(size_t i = 0; i < layerCount; i++) {
        std::cout << "Layer " << i << "("<< _filterCounts[i] << "," << _filterCounts[i+1] <<")\n";
        if(i == layerCount-1) {
            layers.emplace_back(A, _filterCounts[i], _filterCounts[i+1], &softmax);
        } else {
            layers.emplace_back(A, _filterCounts[i], _filterCounts[i+1], &relu);
        }
    }
}

void Model::forwardPass(const Matrix &x) {
    auto* out = &x;
    for(size_t i = 0; i < layerCount; i++) {
        layers[i].forward(*out);
        out = &(layers[i].out);
    }
}

void Model::fit(Matrix& x, const Matrix& y, const Matrix& train, const size_t epochs, const float learningRate,
                const float weightDecay) {
    size_t nTraining = 140;

    for(size_t epoch = 0; epoch < epochs; epoch++) {
        //Forward Pass
        TIC(tic_forw);
        forwardPass(x);
        TOC(tic_forw, "Forward");

        //Backpropagation
        TIC(tic_back);
        //Initital Loss Gradient

        Matrix out = layers[layerCount-1].out;
        Matrix* dz = new Matrix(out.rows, out.cols);
        bli_setm(&BLIS_ZERO, &(dz->mat));
        for(dim_t i = 0; i < y.rows; i++) {
            float isTraining = train.get(i, 0);

            //Only copy for labelled vertices
            if(isTraining != 0) {
                float trueClass = y.get(i, 0);
                for(dim_t j = 0; j < dz->cols; j++) {
                    float currentVal = out.get(i,j);
                    if(j == trueClass) {
                        currentVal -= 1;
                    }
                    dz->set(i,j, currentVal/nTraining);
                }
            }
        }

        //Propagate through hidden layers
        for(size_t i = layerCount-1; i > 0; i--) {
            layers[i].updateWeights(*dz, layers[i-1].out, learningRate, weightDecay);
            dz = layers[i].backpropagate(dz);
        }
        //Propagate through input layer
        layers[0].updateWeights(*dz, x, learningRate, weightDecay);
        TOC(tic_back, "Backprop");


        auto pred = classify(layers[layerCount-1].out);
        float validationAcc = accuracy(y, pred);
        float traininingAcc = trainAccuracy(train, y,pred);

        std::cout << std::left <<"Epoch: " << std::setw(8) << epoch+1 << "Train Acc: " << std::setw(12) << traininingAcc << "Val Acc: " << std::setw(12) << validationAcc << std::endl;

        delete dz;
    }
}

//Transform a NxC Matrix of class probabilities to a Nx1 Matrix containing the predicted classes
Matrix Model::classify(const Matrix &out) {
    auto result = Matrix(out.rows, 1);

    for (size_t v = 0; v < result.rows; v++) {
        float max = 0;
        size_t maxI = 0;
        for (size_t i = 0; i < out.cols; i++) {
            float current = out.get(v,i);
            if (current > max) {
                max = current;
                maxI = i;
            }
        }
        result.set(v, 0, maxI);
    }
    return result;
}

float Model::accuracy(const Matrix &y, const Matrix &y_predicted) {
    size_t correct = 0;
    for(size_t v = 0; v < y.rows; v++) {
        if(y_predicted.get(v,0) == y.get(v,0)) {
            correct++;
        }
    }
    return ((float)correct)/ y.rows;
}

float Model::trainAccuracy(const Matrix &train, const Matrix &y, const Matrix &y_predicted) {
    size_t correct = 0;
    size_t trainSize = 0;
    for(size_t v = 0; v < y.rows; v++) {
        if(train.get(v,0) == 0)
            continue;
        if(y_predicted.get(v,0) == y.get(v,0)) {
            correct++;
        }
        trainSize++;
    }
    return ((float)correct)/trainSize;
}




