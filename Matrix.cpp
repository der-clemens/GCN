//
// Created by Clemens Hartmann on 20/09/2023.
//

#include "Matrix.h"

Matrix::Matrix(size_t _rows, size_t _cols) : rows(_rows), cols(_cols) {
    bli_obj_create(BLIS_FLOAT, _rows, _cols, 1, _rows, &mat);
    bli_setm(&BLIS_ZERO, &mat);
}

Matrix::Matrix(const Matrix &m) : rows(m.rows), cols(m.cols) {
    bli_obj_create(BLIS_FLOAT, rows, cols, 1, rows, &mat);
    bli_copym(&(m.mat), &mat);
}

Matrix::Matrix(Matrix &&m) noexcept : rows(m.rows), cols(m.cols), mat(m.mat) {
    bli_obj_free(&(m.mat));
}

Matrix &Matrix::operator=(const Matrix &m) {
    if(this == &m)
        return *this;
    bli_obj_free(&mat);
    rows = m.rows;
    cols = m.cols;
    bli_obj_create(BLIS_FLOAT, rows, cols, 1, rows, &mat);
    bli_copym(&(m.mat), &mat);
    return *this;
}

Matrix &Matrix::operator=(Matrix &&m) noexcept {
    if(this == &m)
        return *this;
    bli_obj_free(&mat);
    rows = m.rows;
    cols = m.cols;
    mat = m.mat;
    m.mat = obj_t();
    return *this;
}

Matrix::~Matrix() {
    bli_obj_free(&mat);
}

Matrix Matrix::dot(const Matrix &a, const Matrix &b) noexcept {
    assert(a.cols == b.rows);

    Matrix m = Matrix(a.rows, b.cols);
    bli_gemm(&BLIS_ONE, &(a.mat), &(b.mat),&BLIS_ZERO, &(m.mat));
    return m;
}

float Matrix::get(size_t i, size_t j) const {
    double _ = 0;
    double val = 0;
    bli_getijm(i, j, &mat, &val, &_);
    return val;
}

void Matrix::gemm(const Matrix& a, const Matrix& b, Matrix& c, float alpha, float beta) {
    obj_t _alpha, _beta;
    bli_obj_create_1x1( BLIS_FLOAT, &_alpha );
    bli_obj_create_1x1( BLIS_FLOAT, &_beta );
    bli_setsc( alpha, 0.0, &_alpha );
    bli_setsc( beta, 0.0, &_beta );
    bli_gemm(&_alpha, &(a.mat), &(b.mat), &_beta, &(c.mat));
    bli_obj_free(&_alpha);
    bli_obj_free(&_beta);
}

void Matrix::set(size_t i, size_t j, float val) {
    bli_setijm(val, 0, i, j, &mat);
}

void Matrix::transpose() {
    bli_obj_toggle_trans(&mat);
    size_t tmp = rows;
    rows = cols;
    cols = tmp;
}







