#ifndef MATRIX_MUL_H
#define MATRIX_MUL_H

#include "ckks_evaluator.cuh"

extern int batchsize;
extern int input_tokens;
extern int dim_model_weight_in;
extern int dim_model_weight_out;
extern int headnumber;

tuple<matrix, matrix, PhantomCiphertext, matrix> client_offline(CKKSEvaluator* ckks);
matrix client_online(CKKSEvaluator* ckks, matrix& input, matrix& R);
tuple<cipherMatrix,double> server_offline(CKKSEvaluator* ckks, matrix& batched_U_cpack, matrix& batched_V_cpack, PhantomCiphertext& D_ct, matrix& weight);
cipherMatrix server_online(CKKSEvaluator* ckks, matrix& X, matrix& weight, cipherMatrix& RW);
vector<vector<PhantomPlaintext>> server_extend_weight(CKKSEvaluator* ckks, const cipherMatrix& cMat_cpack, const matrix& pMat);


matrix create_random_matrix(double max, double min, int row, int col);

cipherMatrix cMat_mul_pMat(CKKSEvaluator* ckks, const cipherMatrix& cipher_left_matrix_input_cpack, const matrix& right_matrix_input_cpack);

cipherMatrix colpack_cMat_mul_cMat(CKKSEvaluator* ckks, const cipherMatrix& left_cpack, const vector<vector<PhantomCiphertext>>& right_cpack_rotates);
vector<PhantomCiphertext> extend_batched_ct_rotates(CKKSEvaluator* ckks, const PhantomCiphertext& ct, int rows);
cipherMatrix diagpack_cMat_mul_cMat(CKKSEvaluator* ckks, cipherMatrix& left_diagpack, const vector<vector<PhantomCiphertext>>& right_cpack_rotates);

#endif