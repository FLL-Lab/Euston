#pragma once

#include <fstream>
#include <iostream>
#include <vector>

#include "ckks_evaluator.cuh"
#include "phantom.h"
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
namespace nexus {
using namespace std;
using namespace phantom;

class MMEvaluator {
 private:
  CKKSEvaluator *ckks = nullptr;

  void enc_compress_ciphertext(vector<double> &values, PhantomCiphertext &ct);
  vector<PhantomCiphertext> decompress_ciphertext(PhantomCiphertext &encrypted);

 public:
  MMEvaluator(CKKSEvaluator &ckks) : ckks(&ckks) {}

  // Helper functions
  inline vector<vector<double>> read_matrix(const std::string &filename, int rows, int cols) {
    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols));
    std::ifstream file(filename);

    if (!file.is_open()) {
      std::cerr << "Can not open file: " << filename << std::endl;
      return matrix;
    }

    std::string line;
    for (int i = 0; i < rows; ++i) {
      if (std::getline(file, line)) {
        std::istringstream iss(line);
        for (int j = 0; j < cols; ++j) {
          if (!(iss >> matrix[i][j])) {
            std::cerr << "read error: " << filename << " (row: " << i << ", column: " << j << ")" << std::endl;
          }
        }
      }
    }

    file.close();
    return matrix;
  }

  inline vector<vector<double>> transpose_matrix(const vector<vector<double>> &matrix) {
    if (matrix.empty()) {
      return {};
    }
    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<std::vector<double>> transposedMatrix(cols, std::vector<double>(rows));

    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        transposedMatrix[j][i] = matrix[i][j];
      }
    }

    return transposedMatrix;
  }

  // Evaluation function
  void matrix_mul(vector<vector<double>> &x, vector<vector<double>> &y, vector<PhantomCiphertext> &res);
  void multiply_power_of_x(PhantomCiphertext &encrypted, PhantomCiphertext &destination, int index);
};
}  // namespace nexus
