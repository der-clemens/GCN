//
// Created by Clemens Hartmann on 22/09/2023.
//

#ifndef GCN_GRAPHMATRIX_H
#define GCN_GRAPHMATRIX_H

#define RSB true

#include <cstddef>
#include <iostream>
#include "Matrix.h"
#include "rsb.h"
#include "Graph.h"

class GraphMatrix {
public:
    size_t v_count;

#if !RSB
    Matrix m;

    explicit GraphMatrix(const Graph& graph): v_count(graph.v_count), m(Matrix(graph.v_count, graph.v_count)) {
        auto dhi = new float[graph.v_count] {0};
        for (size_t i = 0; i < graph.v_count; i++) {
            size_t degree = graph.index[i+1] - graph.index[i];
            dhi[i] = 1.f/(::sqrt(degree+1));
        }

        for(size_t i = 0; i < graph.v_count; i++) {
            size_t edgeCount = graph.index[i+1] - graph.index[i];

            m.set(i, i, dhi[i]*dhi[i]);
            for(size_t j = 0; j < edgeCount; j++) {
                size_t out = graph.edges[graph.index[i] + j];
                m.set(i, out, dhi[i] * dhi[out]);
            }
        }
    }

    [[nodiscard]] Matrix cross(const Matrix& a) const {
        return Matrix::dot(m,a);
    }
#else
    rsb_mtx_t* m;

    explicit GraphMatrix(const Graph& graph): v_count(graph.v_count) {
        auto dhi = new float[graph.v_count] {0};
        for (size_t i = 0; i < graph.v_count; i++) {
            size_t degree = graph.index[i+1] - graph.index[i];
            dhi[i] = 1.f/(::sqrt(degree+1));
        }
        m = rsb_mtx_alloc_from_coo_begin(graph.v_count+graph.e_count, RSB_NUMERICAL_TYPE_FLOAT,
                                         graph.v_count,
                                         graph.v_count,
                                         RSB_FLAG_LOWER_SYMMETRIC,
                                         nullptr);
        for(size_t i = 0; i < graph.v_count; i++) {
            size_t edgeCount = graph.index[i+1] - graph.index[i];
            float* values = new float[edgeCount+1];
            rsb_coo_idx_t* rowIndicies = new rsb_coo_idx_t[edgeCount + 1];
            rsb_coo_idx_t* columnIndicies = new rsb_coo_idx_t[edgeCount+1];

            values[0] = dhi[i] * dhi[i];
            rowIndicies[0] = i;
            columnIndicies[0] = i;
            for(size_t j = 0; j < edgeCount; j++) {
                size_t out = graph.edges[graph.index[i] + j];
                values[j+1] = dhi[i] * dhi[out];
                rowIndicies[j+1] = i;
                columnIndicies[j+1] = out;
            }
            rsb_mtx_set_vals(m, values, rowIndicies, columnIndicies, edgeCount+1, RSB_FLAG_C_INDICES_INTERFACE);
            delete[] values;
            delete[] rowIndicies;
            delete[] columnIndicies;
        }
        rsb_mtx_alloc_from_coo_end(&m);
    }

    [[nodiscard]] Matrix cross(const Matrix& a) const {
        float* alpha  = new float[1] {1};
        float* beta  = new float[1] {0};
        const rsb_coo_idx_t nrhs { static_cast<rsb_coo_idx_t>(a.cols) };
        float* mat = new float[a.rows* a.cols];
        float* res = new float[a.rows* a.cols];

        int k = 0;
        for(int i = 0; i < a.rows; i++) {
            for(int j = 0; j < a.cols; j++) {
                mat[k] = a.get(i,j);
                k++;
            }
        }

        rsb_spmm(RSB_TRANSPOSITION_N, alpha, m, nrhs, RSB_FLAG_WANT_ROW_MAJOR_ORDER, mat,
                 0,beta,res,0);

        Matrix result = Matrix(a.rows, a.cols);
        int k_2 = 0;
        for(int i = 0; i < a.rows; i++) {
            for(int j = 0; j < a.cols; j++) {
                result.set(i, j, res[k_2]);
                k_2++;
            }
        }

        delete[] alpha;
        delete[] beta;
        delete[] mat;
        delete[] res;

        return result;
    }
#endif

};

#endif //GCN_GRAPHMATRIX_H
