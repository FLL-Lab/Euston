#include <seal/seal.h>
#include <seal/util/uintarith.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "myread.h"
#include "gelu.h"
#include "layer_norm.h"
#include "matrix_mul.h"
#include "softmax.h"
#include "utils.h"

using namespace std;
using namespace seal;
using namespace seal::util;
using namespace chrono;

void MM_test();
// void argmax_test();

// Choose a test target here:
int TEST_TARGET_IDX = 2;

vector<string> TEST_TARGETS = {"MatMul", "Argmax", "GELU", "LayerNorm", "SoftMax"};
vector<int> COEFF_MODULI = {58, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 58};
vector<int> MM_COEFF_MODULI = {60, 40, 40, 40, 60};
double SCALE = pow(2.0, 40);

string TEST_TARGET = TEST_TARGETS[TEST_TARGET_IDX];



int main() {
  if (TEST_TARGET == TEST_TARGETS[0]) {
	MM_test();
    return 0;
  }


// 	if (TEST_TARGET == TEST_TARGETS[1]) {
//     argmax_test();
//     return 0;
//   }

  long logN = 15;
  size_t poly_modulus_degree = 1 << logN;

  EncryptionParameters parms(scheme_type::ckks);
  parms.set_poly_modulus_degree(poly_modulus_degree);
  parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, COEFF_MODULI));

  SEALContext context(parms, true, sec_level_type::none);

  KeyGenerator keygen(context);
  SecretKey secret_key = keygen.secret_key();
  PublicKey public_key;
  keygen.create_public_key(public_key);
  RelinKeys relin_keys;
  keygen.create_relin_keys(relin_keys);
  GaloisKeys galois_keys;
  keygen.create_galois_keys(galois_keys);

  Encryptor encryptor(context, public_key);
  CKKSEncoder encoder(context);
  Evaluator evaluator(context, encoder);
  Decryptor decryptor(context, secret_key);

  CKKSEvaluator ckks_evaluator(context, encryptor, decryptor, encoder, evaluator, SCALE, relin_keys, galois_keys);
  GeLUEvaluator gelu_evaluator(ckks_evaluator);
  LNEvaluator ln_evaluator(ckks_evaluator);
  SoftmaxEvaluator softmax_evaluator(ckks_evaluator);

  Plaintext plain_input;
  Ciphertext cipher_input;
  Ciphertext cipher_output;
  vector<double> output;

  /*
      GELU
  */
  if (TEST_TARGET == TEST_TARGETS[2]) {
    matrix gelu_input = loadtxt(dname+"/input/gelu_input_batched_32768X3072.txt");
	matrix gelu_output = loadtxt(dname+"/output/gelu_output_batched_32768X3072.txt");

    vector<Ciphertext> cipher_gelu_input(gelu_input.size());
	vector<Ciphertext> cipher_gelu_output(gelu_output.size());
	int test_rows = 1;
	for(int i = 0; i < test_rows; i++)
	{
		ckks_evaluator.encoder->encode(gelu_input[i], SCALE, plain_input);
		ckks_evaluator.encryptor->encrypt(plain_input, cipher_gelu_input[i]);
	}

    auto start = high_resolution_clock::now();
	for(int i = 0; i < test_rows; i++)
	{
    	gelu_evaluator.gelu(cipher_gelu_input[i], cipher_gelu_output[i]);
	}
    auto end = high_resolution_clock::now();

    cout << "[GELU] 1 x 3072 takes: " << duration_cast<milliseconds>(end - start).count() << " milliseconds" << endl;
	double errors = 0;
	for(int i = 0; i < test_rows; i++)
	{
		errors += ckks_evaluator.calculate_errors(cipher_gelu_output[i], gelu_output[i]);
	}
	cout << fixed << setprecision(10) << "Average errors: " << errors / test_rows << endl;

    return 0;
  }

  /*
      LayerNorm
  */
  if (TEST_TARGET == TEST_TARGETS[3]) {
    matrix LN_input = loadtxt(dname+"/input/LayerNorm_input_batched_32768X768.txt");
	matrix LN_output = loadtxt(dname+"/output/LayerNorm_output_batched_32768X768.txt");
	vector<Ciphertext> cipher_LN_input(LN_input.size());
	int test_rows = 1;
	for(int i = 0; i < test_rows; i++)
	{
		ckks_evaluator.encoder->encode(LN_input[i], SCALE, plain_input);
		ckks_evaluator.encryptor->encrypt(plain_input, cipher_LN_input[i]);
	}
	vector<Ciphertext> cipher_LN_output(LN_output.size());
    auto start = high_resolution_clock::now();
	for(int i = 0; i < test_rows; i++)
	{
    	ln_evaluator.layer_norm(cipher_LN_input[i], cipher_LN_output[i], 4096);
	}
    auto end = high_resolution_clock::now();

    cout << "[LayerNorm] 1 x 768 takes: " << duration_cast<milliseconds>(end - start).count() << " milliseconds" << endl;
    double errors = 0;
	for(int i = 0; i < test_rows; i++)
	{
		errors += ckks_evaluator.calculate_errors(cipher_LN_output[i], LN_output[i]);
	}
	cout << fixed << setprecision(10) << "Average errors: " << errors / test_rows << endl;

    return 0;
  }

  /*
      Softmax
  */
  if (TEST_TARGET == TEST_TARGETS[4]) {

    matrix softmax_input = loadtxt(dname+"/input/Softmax_input_batched_32768X128.txt");
	matrix softmax_output = loadtxt(dname+"/output/Softmax_output_batched_32768X128.txt");
	int test_rows = 1;
	vector<Ciphertext> cipher_softmax_input(softmax_input.size());
	for(int i = 0; i < test_rows; i++)
	{
		ckks_evaluator.encoder->encode(softmax_input[i], SCALE, plain_input);
		ckks_evaluator.encryptor->encrypt(plain_input, cipher_softmax_input[i]);
	}
	vector<Ciphertext> cipher_softmax_output(softmax_input.size());
    auto start = high_resolution_clock::now();
	for(int i = 0; i < test_rows; i++)
	{
    	softmax_evaluator.softmax(cipher_softmax_input[i], cipher_softmax_output[i], 128);
	}
    auto end = high_resolution_clock::now();

    cout << "[Softmax] 1 x 128 takes: " << duration_cast<milliseconds>(end - start).count()   << " milliseconds" << endl;

	double errors = 0;
	for(int i = 0; i < test_rows; i++)
	{
		errors += ckks_evaluator.calculate_errors(cipher_softmax_output[i], softmax_output[i]);
	}
	cout << fixed << setprecision(10) << "Average errors: " << errors / test_rows << endl;

    return 0;
  }
}

void MM_test() {
  long logN = 13;
  size_t poly_modulus_degree = 1 << logN;

  EncryptionParameters parms(scheme_type::ckks);
  parms.set_poly_modulus_degree(poly_modulus_degree);
  parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, MM_COEFF_MODULI));

  SEALContext context(parms, true, sec_level_type::none);

  KeyGenerator keygen(context);
  SecretKey secret_key = keygen.secret_key();
  PublicKey public_key;
  keygen.create_public_key(public_key);
  RelinKeys relin_keys;
  keygen.create_relin_keys(relin_keys);
  GaloisKeys galois_keys;

  vector<uint32_t> rots;
  for (int i = 0; i < logN; i++) {
    rots.push_back((poly_modulus_degree + exponentiate_uint(2, i)) / exponentiate_uint(2, i));
  }
  keygen.create_galois_keys(rots, galois_keys);

  Encryptor encryptor(context, public_key);
  CKKSEncoder encoder(context);
  Evaluator evaluator(context, encoder);
  Decryptor decryptor(context, secret_key);

  CKKSEvaluator ckks_evaluator(context, encryptor, decryptor, encoder, evaluator, SCALE, relin_keys, galois_keys);
  MMEvaluator mme(ckks_evaluator);

  matrix matrix_input = loadtxt(dname+"/input/IWMM_BERT_input_32x128X768.txt");
  matrix matrix_weight = loadtxt(dname+"/input/IWMM_BERT_weight_768x64.txt");
//   matrix matrix_input = loadtxt(dname+"/input/IWMM_GPT_input_32x128X1600.txt");
//   matrix matrix_weight = loadtxt(dname+"/input/IWMM_GPT_weight_1600x64.txt");
//   matrix matrix_input = loadtxt(dname+"/input/IWMM_LLAMA_input_32x128X4096.txt");
//   matrix matrix_weight = loadtxt(dname+"/input/IWMM_LLAMA_weight_4096X128.txt");

  auto matrix_input_T = mme.transposeMatrix(matrix_input);
  auto matrix_weight_T = mme.transposeMatrix(matrix_weight);

  vector<Ciphertext> res;
  mme.matrix_mul(matrix_input_T, matrix_weight_T, res);  

  matrix matrix_output = loadtxt(dname+"/output/IWMM_BERT_output_32x128X64.txt");
//   matrix matrix_output = loadtxt(dname+"/output/IWMM_GPT_output_32x128X64.txt");
//   matrix matrix_output = loadtxt(dname+"/output/IWMM_LLAMA_output_32x128X128.txt");

  auto matrix_output_T = mme.transposeMatrix(matrix_output);
}


