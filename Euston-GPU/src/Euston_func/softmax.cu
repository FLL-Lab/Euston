#include "softmax.cuh"


matrix diagpack_matrix(const matrix& input)
{
	int rows = input.size();
	int cols = input[0].size();
	matrix output(cols, tensor(rows));
	for(int j = 0; j < cols; j++)
	{
		tensor temp(rows);
		for(int i = 0; i < rows; i++)
		{
			temp[i] = input[i][(i+j) % cols];
		}
		output[j] = temp;
	}
	return output;
}
cipherMatrix cipher_softmax(CKKSEvaluator* ckks, const cipherMatrix& input)
{
	// ckks->print_chain_index(input[0], "input");
	// ckks->print_dec(input[0], "input[0]", 10);
	Rectime func_timer;
	func_timer.start();
	cipherMatrix exp_input = input;
	// #pragma omp parallel for num_threads(32)
	// for(int i = 0; i < input.size(); i++)
	// {
	// 	ckks->evaluator->add_const_inplace(exp_input[i], -1);
	// }
	// ckks->print_dec(exp_input[0], "exp_input[0]",10);
	
	// matrix dec_input(input.size());
	// matrix real_exp(input.size(), tensor(4096));

	for(int i = 0; i < input.size(); i++)
	{
		exp_input[i] = cipher_exp_5(ckks, exp_input[i]);
		// Plaintext pt;
		// ckks->decryptor->decrypt(input[i], pt);
		// ckks->encoder->decode(pt, dec_input[i]);
		// for(int j = 0; j < dec_input[i].size(); j++)
		// {
		// 	real_exp[i][j] = exp(dec_input[i][j]);
		// }
	}
	// tensor sum_vector(real_exp[0].size(), 0.0);
	// for (int j = 0; j < real_exp.size(); j++) {
	// 	for (int i = 0; i < sum_vector.size(); i++) {
	// 		sum_vector[i] += real_exp[j][i];
	// 	}
	// }
	// for(int i = 0; i < 10; i++)
	// {
	// 	cout << sum_vector[i] << " ";
	// }
	// cout << endl;
	// ckks->print_chain_index(exp_input[0], "exp_input[0]");
	// ckks->print_dec(exp_input[0], "exp_input[0]", 10);
	PhantomCiphertext exp_sum;
	ckks->evaluator->add_many(exp_input, exp_sum);
	// ckks->print_dec(exp_sum, "exp_sum", 10);

	// // Find the max value in exp_sum
	// Plaintext plain_exp_sum;
	// ckks->decryptor->decrypt(exp_sum, plain_exp_sum);
	// vector<double> decoded_exp_sum;
	// ckks->encoder->decode(plain_exp_sum, decoded_exp_sum);
	// double max_value = *max_element(decoded_exp_sum.begin(), decoded_exp_sum.end());
	// std::cout << "Max value in decrypted exp_sum: " << max_value << std::endl;
	
	// tensor real_exp_sum_inv(sum_vector.size());
	// for(int i = 0; i < sum_vector.size();i++)
	// {
	// 	real_exp_sum_inv[i] = 1 / sum_vector[i];
	// }
	// for(int i = 0; i < 10; i++)
	// {
	// 	cout << real_exp_sum_inv[i] << " ";
	// }
	// cout << endl;
	PhantomCiphertext inv_sum = cipher_inverse(ckks, exp_sum, pow(2,5));
	// ckks->print_chain_index(inv_sum, "inv_sum");
	// ckks->print_dec(inv_sum, "inv_sum", 10);


	// matrix real_softmax(real_exp.size());
	cipherMatrix softmax_output(exp_input.size());
	auto pram_id = inv_sum.params_id( );
	// #pragma omp parallel for num_threads(32)
	for(int i = 0; i < input.size(); i++)
	{
		// real_softmax[i] = hadamard_product(real_exp[i], real_exp_sum_inv);
		// for(int j = 0; j < 10; j++)
		// {
		// 	cout << real_softmax[i][j] << " ";
		// }
		// cout << endl;
		ckks->evaluator->mod_switch_to_inplace(exp_input[i], pram_id);
		ckks->evaluator->multiply(exp_input[i], inv_sum, softmax_output[i]);
		ckks->evaluator->relinearize_inplace(softmax_output[i], *(ckks->relin_keys));
		ckks->evaluator->rescale_to_next_inplace(softmax_output[i]);
		
		// ckks->print_dec(softmax_output[i], "softmax_output[i]", 10);
	}
	func_timer.elapsed("cipher_softmax");
	// ckks->print_chain_index(exp_input[0], "exp_input");
	// ckks->print_dec(exp_input[0], "exp_input[0]", 10);
	return softmax_output;
}