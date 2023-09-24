//
// Created by Clemens Hartmann on 21/09/2023.
//

#include "Layer.h"

void d_relu(Matrix& dz, const Matrix& in) {
    assert(dz.rows == in.rows);
    assert(dz.cols == in.cols);

    for(dim_t row_idx = 0; row_idx < dz.rows; row_idx++) {
        for(dim_t col_idx = 0; col_idx < dz.cols; col_idx++) {
            float ele = in.get(row_idx, col_idx);
            if(ele <= 0) {
                dz.set(row_idx, col_idx, 0);
            }
        }
    }
}

Layer::Layer(GraphMatrix &_A, size_t _inFeatures, size_t _outFeatures, void (*_activation)(Matrix&))
    : A(_A),
    activation(_activation),
    w(Matrix(_inFeatures, _outFeatures)),
    out(Matrix(_A.v_count, _outFeatures))
    {
    bli_randm(&(w.mat));
    bli_setm(&BLIS_ZERO, &(out.mat));
}

void Layer::forward(const Matrix& x) {
    out = Matrix::dot(x,w);

    out = A.cross(out);
    activation(out);
}

Matrix* Layer::backpropagate(Matrix* dz) {
    d_relu(*dz, out);
    w.transpose();
    auto dz_out = new Matrix(Matrix::dot(*dz, w));
    w.transpose();

    dz_out = new Matrix(A.cross(*dz_out));
    delete dz;
    return dz_out;
}
//W = W - a∂W - adW => W = (1-ad)W - a∂W
void Layer::updateWeights(const Matrix& dz, Matrix& prevIn, float learningRate, float weightDecay) {
    prevIn.transpose();
    Matrix::gemm(prevIn, dz, w, -learningRate, (1-learningRate*weightDecay));
    prevIn.transpose();
}
