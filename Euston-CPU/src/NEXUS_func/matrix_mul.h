#pragma once

#include <seal/seal.h>
#include <Eigen/Dense>
#include <vector>
#include "utils.h"
#include "ckks_evaluator.h"

#define BERT

#ifdef BERT
#define batchsize 32
#define input_tokens 128
#define dim_model_weight_in 768
#define dim_model_weight_out 64
#define headnumber 12
#endif

#ifdef GPT
#define batchsize 128
#define input_tokens 32
#define dim_model_weight_in 1600
#define dim_model_weight_out 64
#define headnumber 25
#endif

#ifdef LLAMA
#define batchsize 512
#define input_tokens 8
#define dim_model_weight_in 4096
#define dim_model_weight_out 128
#define headnumber 32
#endif

class MMEvaluator {
 private:
  CKKSEvaluator *ckks = nullptr;

  void enc_compress_ciphertext(vector<double> &vec, Ciphertext &ct);
  void multiply_power_of_x(Ciphertext &encrypted, Ciphertext &destination, int index);
  vector<Ciphertext> expand_ciphertext(const Ciphertext &encrypted, uint32_t m, GaloisKeys &galkey, vector<uint32_t> &galois_elts);

 public:
  MMEvaluator(CKKSEvaluator &ckks) {
    this->ckks = &ckks;
  }
	matrix readEmbeddings(string dir_name, int number);
  void matrix_mul(matrix &x, matrix &y, vector<Ciphertext> &res);
	
  matrix readMatrix(const string &filename, int rows, int cols);
  matrix transposeMatrix(const matrix &mat);
};
