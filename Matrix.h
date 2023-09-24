//
// Created by Clemens Hartmann on 20/09/2023.
//

#ifndef GCN_MATRIX_H
#define GCN_MATRIX_H


#include <cassert>
#include "blis.h"

class Matrix {
public:
    obj_t mat;
    size_t rows;
    size_t cols;

    Matrix(size_t _rows, size_t _cols);
    Matrix(const Matrix& m);
    Matrix(Matrix&& m) noexcept;
    Matrix& operator=(const Matrix& m);
    Matrix& operator=(Matrix&& m) noexcept;

    ~Matrix();

    [[nodiscard]] static Matrix dot(const Matrix& a, const Matrix& b) noexcept;
    static void gemm(const Matrix& a, const Matrix& b, Matrix& c, float alpha, float beta);
    void transpose();

    [[nodiscard]] float get(size_t i, size_t j) const;
    void set(size_t i, size_t j, float val);


};




#endif //GCN_MATRIX_H
