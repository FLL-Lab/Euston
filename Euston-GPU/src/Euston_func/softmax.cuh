#ifndef SOFTMAX_H
#define SOFTMAX_H


#include "ckks_evaluator.cuh"


matrix diagpack_matrix(const matrix& input);
cipherMatrix cipher_softmax(CKKSEvaluator* ckks, const cipherMatrix& input);

#endif