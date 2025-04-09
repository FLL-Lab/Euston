#ifndef CKKS_EVALUATOR_H
#define CKKS_EVALUATOR_H

#include "utils.h"
#include <seal/seal.h>
#include <seal/util/uintarith.h>
#include <seal/util/polyarithsmallmod.h>

using namespace seal;
using namespace seal::util;

using cipherMatrix = vector<Ciphertext>;
extern vector<int> COEFF_MODULI;
extern vector<int> MATRIX_COEFF_MODULI;
extern double SCALE;

class CKKSEvaluator 
{
public:
	double sgn_factor = 0.5;
	vector<double> F4_COEFFS = {0, 315, 0, -420, 0, 378, 0, -180, 0, 35};
	// should be divided by (1 << 7)
	int f4_scale = (1 << 7);
	vector<double> G4_COEFFS = {0, 5850, 0, -34974, 0, 97015, 0, -113492, 0, 46623};
	// should be divided by (1 << 10)
	int g4_scale = (1 << 10);
public:
	shared_ptr<SEALContext> context;
	shared_ptr<Encryptor> encryptor;
	shared_ptr<Decryptor> decryptor;
	shared_ptr<CKKSEncoder> encoder;
	shared_ptr<Evaluator> evaluator;
	shared_ptr<RelinKeys> relin_keys;
	shared_ptr<GaloisKeys> galois_keys;
	shared_ptr<GaloisKeys> galois_keys_allrotate;
	double scale;
	size_t N;
	size_t slot_count;
	size_t degree;
	size_t comm = 0;
	size_t round = 0;
	vector<uint32_t> rots;

	CKKSEvaluator(shared_ptr<SEALContext> context, shared_ptr<Encryptor> encryptor,shared_ptr<Decryptor> decryptor, shared_ptr<CKKSEncoder> encoder, shared_ptr<Evaluator> evaluator, shared_ptr<RelinKeys> relin_keys, shared_ptr<GaloisKeys> galois_keys,  shared_ptr<GaloisKeys> galois_keys_allrotate, double scale) 
		: context(context), encryptor(encryptor), decryptor(decryptor), encoder(encoder), evaluator(evaluator),
	relin_keys(relin_keys), galois_keys(galois_keys), galois_keys_allrotate(galois_keys_allrotate), scale(scale)
	{
		N = encoder->slot_count() * 2;
		degree = N;
		slot_count = encoder->slot_count();

		for (int i = 0; i < uint(ceil(log2(degree))); i++) 
		{
			rots.push_back((degree + exponentiate_uint(2, i)) / exponentiate_uint(2, i));
		}
	}

	void print_chain_index(Ciphertext ct, string ct_name);
	void print_dec(Ciphertext ct, string des, int num);
	double calculate_errors(Ciphertext ct, tensor list);

};

CKKSEvaluator construct_ckks_evaluator_matrix(vector<int> coeffs);
CKKSEvaluator construct_ckks_evaluator_nonlinear(long logN, vector<int> coeffs);
void switch_ckks_evaluator(Ciphertext& ct, CKKSEvaluator* ckks1, CKKSEvaluator* ckks2);
void encrypting_compress(const CKKSEvaluator* ckks, const tensor& list, Ciphertext &ct);
vector<Ciphertext> encrypting_decompress(const CKKSEvaluator* ckks, const Ciphertext &encrypted, uint32_t m, GaloisKeys &galkey, const vector<uint32_t> &galois_elts);
void multiply_power_of_x(const CKKSEvaluator* ckks, Ciphertext &encrypted, Ciphertext &destination, int index);
Ciphertext cipher_exp(CKKSEvaluator* ckks, Ciphertext& x);
Ciphertext cipher_exp_5(CKKSEvaluator* ckks, Ciphertext& x);
Ciphertext cipher_inverse(CKKSEvaluator* ckks, Ciphertext x, double scl, int iter=4);
Ciphertext cipher_sqrt(CKKSEvaluator* ckks, Ciphertext x);

uint64_t get_modulus(CKKSEvaluator* ckks, Ciphertext &x, int k);
void eval_odd_deg9_poly(CKKSEvaluator* ckks, vector<double> &a, Ciphertext &x, Ciphertext &dest);
Ciphertext cipher_sign(CKKSEvaluator* ckks, Ciphertext x, int d_g, int d_f, double sgn_factor = 0.5);
double cal_ct_size(Ciphertext ct);
double cal_pt_size(matrix pt);
#endif