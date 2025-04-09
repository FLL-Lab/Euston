#include "gelu.cuh"

// gelu(x) = x / (1+exp(-1.702*x))
// x / (1+exp(-1.702*x)) = x*exp(-1.702*minBound) / (exp(-1.702*minBound)+exp(-1.702*(x+minBound))
cipherMatrix cipher_gelu(CKKSEvaluator* ckks, const cipherMatrix& input) 
{
	cipherMatrix output(input.size());
	#pragma omp parallel for num_threads(32)
	for (int i = 0; i < input.size(); i++) 
	{
		PhantomCiphertext y = input[i];
		PhantomCiphertext numerator = y;
		PhantomCiphertext denominator;
		ckks->evaluator->multiply_const_inplace(y, -1.702);
		ckks->evaluator->rescale_to_next_inplace(y);
		// ckks->print_chain_index(y, "y");
		// ckks->print_dec(y, "y", 5);
		denominator = cipher_exp_5(ckks, y);
		// ckks->print_chain_index(denominator, "denominator");
		// ckks->print_dec(denominator, "denominator", 5);
		ckks->evaluator->add_const_inplace(denominator, 1);
		// ckks->print_dec(denominator, "denominator", 5);
		denominator = cipher_inverse(ckks, denominator, 1, 3);
		// ckks->print_dec(denominator, "denominator", 5);
		ckks->evaluator->mod_switch_to_inplace(numerator, denominator.params_id());
		ckks->evaluator->multiply(numerator, denominator, output[i]);
		ckks->evaluator->relinearize_inplace(output[i], *(ckks->relin_keys));
		ckks->evaluator->rescale_to_next_inplace(output[i]);
	}
	return output;
} 

cipherMatrix cipher_gelu_rectify(CKKSEvaluator* ckks, const cipherMatrix& input, const cipherMatrix& sign)
{
	cipherMatrix rectify(input.size());
	#pragma omp parallel for num_threads(32)
	for(int i = 0; i < input.size(); i++)
	{
		PhantomCiphertext ct;
		ckks->evaluator->multiply_const(sign[i], 1.702/2, ct);
		ckks->evaluator->rescale_to_next_inplace(ct);
		ckks->evaluator->multiply_inplace(ct, input[i]);
		ckks->evaluator->relinearize_inplace(ct, *(ckks->relin_keys));
		ckks->evaluator->rescale_to_next_inplace(ct);

		PhantomCiphertext ct1;
		ckks->evaluator->multiply_const(input[i], -1.702/2, ct1);
		ckks->evaluator->rescale_to_next_inplace(ct1);

		// ckks->print_chain_index(ct1, "ct1");
		// ckks->print_chain_index(ct, "ct");
		ct.scale() = ct1.scale();
		ckks->evaluator->add_inplace(ct, ct1);
		
		// ckks->print_dec(ct, "b:ct", 5);
		rectify[i] = cipher_exp_5(ckks, ct);
		// ckks->print_dec(rectify[i], "rectify", 5);
		PhantomCiphertext ct_sign = sign[i];
		ckks->evaluator->mod_switch_to_inplace(ct_sign, rectify[i].params_id());
		ckks->evaluator->multiply_inplace(rectify[i], ct_sign);
		ckks->evaluator->relinearize_inplace(rectify[i], *(ckks->relin_keys));
		ckks->evaluator->rescale_to_next_inplace(rectify[i]);
		// ckks->print_dec(rectify[i], "c: rectify", 5);
	}
	return rectify;
}