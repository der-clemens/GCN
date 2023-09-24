//
// Created by Clemens Hartmann on 25/08/2023.
//

#include "rsb.h"
#include <vector>
#include <stdexcept>

int main() {
    const rsb_nnz_idx_t nnzA { 7 };
    const rsb_coo_idx_t nrA { 6 }, ncA { 6 }, nrhs { 1 };
    const std::vector<rsb_coo_idx_t> IA {0,1,2,3,4,5,1}, JA {0,1,2,3,4,5,0};
    const std::vector<RSB_DEFAULT_TYPE> VA {1,1,1,1,1,1,2}, X(ncA,1);
    std::vector<RSB_DEFAULT_TYPE> Y(nrA,0);
    const RSB_DEFAULT_TYPE alpha {2}, beta {1};
    const rsb_type_t typecode = RSB_NUMERICAL_TYPE_DEFAULT; // see rsb_types.h

    if(rsb_lib_init(RSB_NULL_INIT_OPTIONS) != RSB_ERR_NO_ERROR)
        throw  std::runtime_error("failure running rsb_lib_init!");

    auto mtxAp = rsb_mtx_alloc_from_coo_const(
            VA.data(),IA.data(),JA.data(),nnzA,typecode,nrA,ncA,1,1,RSB_FLAG_NOFLAGS|RSB_FLAG_DUPLICATES_SUM, nullptr);
    if ( ! mtxAp)
        throw  std::runtime_error("failure running rsb_mtx_alloc_from_coo_const!");
    rsb_err_t errval = rsb_spmm(RSB_TRANSPOSITION_N,&alpha,mtxAp,nrhs,RSB_FLAG_WANT_COLUMN_MAJOR_ORDER,X.data(),ncA,&beta,Y.data(),nrA);
    if ( errval != RSB_ERR_NO_ERROR )
        throw  std::runtime_error("failure running rsb_spmm!");
}