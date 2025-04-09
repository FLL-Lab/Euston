#ifndef LAYERNORM_H
#define LAYERNORM_H

#include "ckks_evaluator.cuh"

cipherMatrix cipher_layernorm(CKKSEvaluator* ckks, const cipherMatrix& input);

#endif