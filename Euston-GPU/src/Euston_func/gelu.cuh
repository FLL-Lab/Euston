#ifndef GELU_H
#define GELU_H

#include "ckks_evaluator.cuh"

cipherMatrix cipher_gelu(CKKSEvaluator* ckks, const cipherMatrix& input);
cipherMatrix cipher_gelu_rectify(CKKSEvaluator* ckks, const cipherMatrix& input, const cipherMatrix& sign);

#endif // GELU_H