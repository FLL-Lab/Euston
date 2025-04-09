#ifndef LAYERNORM_H
#define LAYERNORM_H

#include "ckks_evaluator.h"

cipherMatrix cipher_layernorm(CKKSEvaluator* ckks, const cipherMatrix& input);

#endif