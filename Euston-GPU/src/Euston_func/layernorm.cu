#include "layernorm.cuh"

cipherMatrix cipher_layernorm(CKKSEvaluator* ckks,const cipherMatrix& input)
{
	cipherMatrix output(input.size());
	cipherMatrix cp_input = input;
	cipherMatrix x(input.size());	
	int siz = input.size();
	#pragma omp parallel for num_threads(32)
	for(int i = 0; i < cp_input.size(); i++)
	{
		vector<PhantomCiphertext> x_elems(cp_input.size());
		for(int j = 0; j < cp_input.size(); j++)
		{
			ckks->evaluator->sub(cp_input[i], cp_input[j], x_elems[j]);
		}
		ckks->evaluator->add_many(x_elems, x[i]);
		x[i].scale() *= siz;
	}
	// ckks->print_chain_index(x[0], "x");
	PhantomCiphertext denominator;
	vector<PhantomCiphertext> x_squares(cp_input.size());
	#pragma omp parallel for num_threads(32)
	for(int i = 0; i < cp_input.size(); i++)
	{
		ckks->evaluator->multiply_reduced_error(x[i], x[i], *(ckks->relin_keys), x_squares[i]);
		// ckks->evaluator->relinearize_inplace(x_squares[i], *(ckks->relin_keys));
		ckks->evaluator->rescale_to_next_inplace(x_squares[i]);
	}
	ckks->evaluator->add_many(x_squares, denominator);
	// ckks->print_chain_index(denominator, "denominator");
	//测试一下最大值最小值
	// Plaintext pt;
	// tensor ten;
	// ckks->decryptor->decrypt(denominator, pt);
	// ckks->encoder->decode(pt, ten);
	// double max_val = *max_element(ten.begin(), ten.end());
	// double min_val = *min_element(ten.begin(), ten.end());
	// cout << "max_val: " << max_val << endl;
	// cout << "min_val: " << min_val << endl;
	
	int magnify = pow(2,10);
	denominator.scale() *= magnify; // scale to [0,2] 
	// ckks->print_chain_index(denominator, "denominator");
	// ckks->print_dec(denominator, "denominator", 10);
	ckks->evaluator->add_const_inplace(denominator, -1.0); // variance^2 = (1+deno)^2
	denominator = cipher_sqrt(ckks, denominator);
	// ckks->print_chain_index(denominator, "denominator");
	// ckks->print_dec(denominator, "denominator_sqrt", 10);

	// ckks->decryptor->decrypt(denominator, pt);
	// ckks->encoder->decode(pt, ten);
	// max_val = *max_element(ten.begin(), ten.end());
	// min_val = *min_element(ten.begin(), ten.end());
	// cout << "max_val: " << max_val << endl;
	// cout << "min_val: " << min_val << endl;

	
	denominator = cipher_inverse(ckks, denominator, 2);
	// ckks->print_dec(denominator, "denominator_inverse", 10);
                                                                                                                                                       
	#pragma omp parallel for num_threads(32)
	for(int i = 0; i < cp_input.size(); i++)
	{
		x[i].scale() *= sqrt(magnify);
		x[i].scale() /= sqrt(cp_input.size());
		ckks->evaluator->mod_switch_to_inplace(x[i], denominator.params_id());
		ckks->evaluator->multiply(x[i], denominator, output[i]);
		ckks->evaluator->relinearize_inplace(output[i], *(ckks->relin_keys));
		ckks->evaluator->rescale_to_next_inplace(output[i]);
	}

	return output;
}