#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// #include "argmax.cuh"
#include "ckks_evaluator.cuh"
#include "gelu.cuh"
#include "layer_norm.cuh"
#include "matrix_mul.cuh"
#include "phantom.h"
#include "softmax.cuh"
#include "nexus_utils.cuh"
#include "myread.cuh"
#include "utils.cuh"
using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;
using namespace nexus;

// Choose test target here:
int TEST_TARGET_IDX = 3;

size_t N = 1ULL << 13;
size_t MM_LOG_N = 13;
size_t MM_N = 1ULL << MM_LOG_N;

double SCALE = pow(2.0, 40);

vector<string> TEST_TARGETS = {"MatMul", "Argmax", "SoftMax", "LayerNorm", "GELU"};
vector<vector<int>> COEFF_MODULI =
    {
        {60, 40, 60},                                                                      // MatMul (0)
        {17},                                                                              // Argmax (1) - Number of Moduli
        {58, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 58},          // SoftMax (2)
        {58, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 58},  // LayerNorm (3)
        {58, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 58}   // GELU (4)
};

string TEST_TARGET = TEST_TARGETS[TEST_TARGET_IDX];
vector<int> TEST_COEFF_MODULI = COEFF_MODULI[TEST_TARGET_IDX];

void MM_test() {
  EncryptionParameters parms(scheme_type::ckks);

  parms.set_poly_modulus_degree(MM_N);
  parms.set_coeff_modulus(CoeffModulus::Create(MM_N, TEST_COEFF_MODULI));

  PhantomContext context(parms);
  PhantomCKKSEncoder encoder(context);

  PhantomSecretKey secret_key(context);
  PhantomPublicKey public_key = secret_key.gen_publickey(context);
  PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
  PhantomGaloisKey galois_keys;

  std::vector<std::uint32_t> galois_elts;
  for (int i = 0; i < MM_LOG_N; i++) {
    galois_elts.push_back((MM_N + exponentiate_uint(2, i)) / exponentiate_uint(2, i));
  }

  CKKSEvaluator ckks_evaluator(&context, &public_key, &secret_key, &encoder, &relin_keys, &galois_keys, SCALE, galois_elts);
  ckks_evaluator.decryptor.create_galois_keys_from_elts(galois_elts, *(ckks_evaluator.galois_keys));

  MMEvaluator mme(ckks_evaluator);

  matrix matrix_input = loadtxt(dname+"/input/IWMM_BERT_input_32x128X768.txt");
  matrix matrix_weight = loadtxt(dname+"/input/IWMM_BERT_weight_768x64.txt");
//   matrix matrix_input = loadtxt(dname+"/input/IWMM_GPT_input_32x128X1600.txt");
//   matrix matrix_weight = loadtxt(dname+"/input/IWMM_GPT_weight_1600x64.txt");
//   matrix matrix_input = loadtxt(dname+"/input/IWMM_LLAMA_input_32x128X4096.txt");
//   matrix matrix_weight = loadtxt(dname+"/input/IWMM_LLAMA_weight_4096X128.txt");

  auto matrix_input_T = mme.transpose_matrix(matrix_input);
  auto matrix_weight_T = mme.transpose_matrix(matrix_weight);


  vector<PhantomCiphertext> res;
  mme.matrix_mul(matrix_input_T, matrix_weight_T, res);
}


int main() {
  if (TEST_TARGET == "MatMul") {
    MM_test();
    return 0;
  }

  EncryptionParameters params(scheme_type::ckks);

  params.set_poly_modulus_degree(N);
  params.set_coeff_modulus(CoeffModulus::Create(N, TEST_COEFF_MODULI));

  PhantomContext context(params);

  PhantomSecretKey secret_key(context);
  PhantomPublicKey public_key = secret_key.gen_publickey(context);
  PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
  PhantomGaloisKey galois_keys = secret_key.create_galois_keys(context);

  PhantomCKKSEncoder encoder(context);

  CKKSEvaluator ckks_evaluator(&context, &public_key, &secret_key, &encoder, &relin_keys, &galois_keys, SCALE);

  vector<double> input;
  PhantomPlaintext plain_input;
  PhantomCiphertext cipher_input;
  PhantomCiphertext cipher_output;
  PhantomPlaintext plain_output;
  vector<double> output;

  /*
    GELU
  */
  if (TEST_TARGET == "GELU") {
	
    GELUEvaluator gelu_evaluator(ckks_evaluator);
	matrix gelu_input = loadtxt(dname+"/input/gelu_input_batched_32768X3072.txt");
	matrix gelu_output = loadtxt(dname+"/output/gelu_output_batched_32768X3072.txt");
	vector<PhantomCiphertext> cipher_gelu_input(gelu_input.size());
	vector<PhantomCiphertext> cipher_gelu_output(gelu_output.size());
	int test_rows = 1;

	for(int i = 0; i < test_rows; i++)
	{
		ckks_evaluator.encoder.encode(gelu_input[i], SCALE, plain_input);
		ckks_evaluator.encryptor.encrypt(plain_input, cipher_gelu_input[i]);
	}

	auto start = high_resolution_clock::now();
	for(int i = 0; i < test_rows; i++)
	{
    	gelu_evaluator.gelu(cipher_gelu_input[i], cipher_gelu_output[i]);
	}
	auto end = high_resolution_clock::now();

    cout << "[GELU] 1 x 3072 takes: " << duration_cast<milliseconds>(end - start).count() << " milliseconds" << endl;
  }

  /*
    LayerNorm
  */
  if (TEST_TARGET == "LayerNorm") {
    LNEvaluator ln_evaluator(ckks_evaluator);
	matrix LN_input = loadtxt(dname+"/input/LayerNorm_input_batched_32768X768.txt");
	matrix LN_output = loadtxt(dname+"/output/LayerNorm_output_batched_32768X768.txt");


    vector<PhantomCiphertext> cipher_LN_input(LN_input.size());
	int test_rows = 1;
	for(int i = 0; i < test_rows; i++)
	{
		ckks_evaluator.encoder.encode(LN_input[i], SCALE, plain_input);
		ckks_evaluator.encryptor.encrypt(plain_input, cipher_LN_input[i]);
	}
	vector<PhantomCiphertext> cipher_LN_output(LN_output.size());
    auto start = high_resolution_clock::now();
	for(int i = 0; i < test_rows; i++)
	{
    	ln_evaluator.layer_norm(cipher_LN_input[i], cipher_LN_output[i], 1024);
	}
    auto end = high_resolution_clock::now();

    cout << "[LayerNorm] 1 x 768 takes: " << duration_cast<milliseconds>(end - start).count() << " milliseconds" << endl;
  }

  /*
    Softmax
  */
  if (TEST_TARGET == "SoftMax") {
    SoftmaxEvaluator softmax_evaluator(ckks_evaluator);
	matrix softmax_input = loadtxt(dname+"/input/Softmax_input_batched_32768X128.txt");
	matrix softmax_output = loadtxt(dname+"/output/Softmax_output_batched_32768X128.txt");
	int test_rows = 1;
	vector<PhantomCiphertext> cipher_softmax_input(softmax_input.size());
	for(int i = 0; i < test_rows; i++)
	{
		ckks_evaluator.encoder.encode(softmax_input[i], SCALE, plain_input);
		ckks_evaluator.encryptor.encrypt(plain_input, cipher_softmax_input[i]);
	}
	vector<PhantomCiphertext> cipher_softmax_output(softmax_input.size());
    auto start = high_resolution_clock::now();
	for(int i = 0; i < test_rows; i++)
	{
    	softmax_evaluator.softmax(cipher_softmax_input[i], cipher_softmax_output[i], 128);
	}
    auto end = high_resolution_clock::now();

    cout << "[Softmax] 1 x 128 takes: " << duration_cast<milliseconds>(end - start).count()   << " milliseconds" << endl;
  }
}
