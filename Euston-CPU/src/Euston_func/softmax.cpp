#include "softmax.h"


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
	Rectime func_timer;
	func_timer.start();
	cipherMatrix exp_input = input;
	#pragma omp parallel for num_threads(32)
	for(int i = 0; i < input.size(); i++)
	{
		exp_input[i] = cipher_exp_5(ckks, exp_input[i]);
	}

	Ciphertext exp_sum;
	ckks->evaluator->add_many(exp_input, exp_sum);

	Ciphertext inv_sum = cipher_inverse(ckks, exp_sum, pow(2,5));

	cipherMatrix softmax_output(exp_input.size());
	auto pram_id = inv_sum.parms_id( );
	#pragma omp parallel for num_threads(32)
	for(int i = 0; i < input.size(); i++)
	{
		ckks->evaluator->mod_switch_to_inplace(exp_input[i], pram_id);
		ckks->evaluator->multiply(exp_input[i], inv_sum, softmax_output[i]);
		ckks->evaluator->relinearize_inplace(softmax_output[i], *(ckks->relin_keys));
		ckks->evaluator->rescale_to_next_inplace(softmax_output[i]);
	}
	return softmax_output;
}