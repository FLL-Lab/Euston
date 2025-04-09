#include "matrix_mul.h"
int batchsize;
int input_tokens;
int dim_model_weight_in;
int dim_model_weight_out;
int headnumber;
matrix create_random_matrix(double max, double min, int row, int col)
{
	static unsigned int counter = 1000; 
    srand(counter++);

    matrix mat(row, vector<double>(col));

    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            double random_value = min + static_cast<double>(rand()) / (static_cast<double>(RAND_MAX / (max - min)));
            mat[i][j] = random_value;
        }
    }

    return mat;
}

tuple<matrix, matrix, Ciphertext, matrix> client_offline(CKKSEvaluator* ckks)
{
	Rectime timer;
	timer.start();
	vector<matrix> Rs(batchsize);
	matrix hat_R;
	for(int i = 0; i < batchsize; i++)
	{
		Rs[i] = create_random_matrix(1, -1, input_tokens, dim_model_weight_in);
		for(int j = 0; j < input_tokens; j++)
		{
			hat_R.push_back(Rs[i][j]);
		}
	}
	vector<matrix> multi_U(batchsize), multi_V(batchsize);
	vector<tensor> multi_D(batchsize);
	#pragma omp parallel for num_threads(32)
	for(int i = 0; i < batchsize; i++)
	{
		svd_decomposition(Rs[i], multi_U[i], multi_D[i], multi_V[i]);
	}
	matrix hat_U, hat_V;
	tensor hat_D;
	for(int i = 0; i < batchsize; i++)
	{
		for(int j = 0; j < multi_U[0].size(); j++)
		{
			hat_U.push_back(multi_U[i][j]);
		}
	}
	for(int i = 0; i < batchsize; i++)
	{
		for(int j = 0; j < multi_V[0].size(); j++)
		{
			hat_V.push_back(multi_V[i][j]);
		}
	}
	for(int i = 0; i < batchsize; i++)
	{
		hat_D.insert(hat_D.end(), multi_D[i].begin(), multi_D[i].end());
	}
	matrix hat_U_cpack = mat_transpose(hat_U);
	// timer.elapsed("--svd_decomposition");
	Ciphertext hat_D_ct;
	timer.start();
	encrypting_compress(ckks, hat_D, hat_D_ct);
	// timer.elapsed("--encrypting_compress");
	
	return make_tuple(hat_U_cpack, hat_V, hat_D_ct, hat_R);
}

matrix client_online(CKKSEvaluator* ckks, matrix& input, matrix& R)
{
	matrix hat_X = mat_sub_mat(input, R);
	return hat_X;
}

tuple<cipherMatrix,double> server_offline(CKKSEvaluator* ckks, matrix& hat_U_cpack, matrix& hat_V, Ciphertext& D_ct, matrix& weight)
{	
	Rectime func_timer;
	func_timer.start();
	vector<Ciphertext> diag_extend_cts = encrypting_decompress(ckks, D_ct, ckks->degree, *ckks->galois_keys, ckks->rots);

	for(auto &ct: diag_extend_cts)
	{
		ct.scale() *= ckks->N;
	}

	// Compute [E] = U * [D]
	func_timer.start();
	int one_U_cols = hat_U_cpack.size();
	int one_U_rows = (int)hat_U_cpack[0].size() / batchsize;
	cipherMatrix E_cpack(one_U_cols);

	vector<Plaintext> batched_onehot_pts(batchsize);
	for(int ell = 0; ell < batchsize; ell++)
	{
		vector<double> onehot;
		onehot.reserve(batchsize*one_U_rows);
		onehot.insert(onehot.end(), ell * one_U_rows, 0.0);
		onehot.insert(onehot.end(), one_U_rows, 1.0);
		onehot.insert(onehot.end(), (batchsize - ell - 1) * one_U_rows, 0.0);
		ckks->encoder->encode(onehot, ckks->scale, batched_onehot_pts[ell]);
	}
	#pragma omp parallel for num_threads(32)
	for(int j = 0; j < one_U_cols; j++)
	{
		vector<Ciphertext> batched_merged_D_ell_cts(batchsize);
		
		for(int ell = 0; ell < batchsize; ell++)
		{
			ckks->evaluator->multiply_plain(diag_extend_cts[ell*one_U_rows+j], batched_onehot_pts[ell], batched_merged_D_ell_cts[ell]);
		}
		Ciphertext batched_merged_D_ct_j;
		ckks->evaluator->add_many(batched_merged_D_ell_cts, batched_merged_D_ct_j);
		ckks->evaluator->rescale_to_next_inplace(batched_merged_D_ct_j);
		
		Plaintext psi;
		ckks->encoder->encode(hat_U_cpack[j], batched_merged_D_ct_j.parms_id(), batched_merged_D_ct_j.scale(), psi);
		ckks->evaluator->multiply_plain(batched_merged_D_ct_j, psi, E_cpack[j]);
		ckks->evaluator->rescale_to_next_inplace(E_cpack[j]);
	}
	double left_time = func_timer.end();

	// Compute H = V * W
	func_timer.start();
	matrix H = mat_mul_mat(hat_V, weight);

	// Extend H to merged replicated vectors
	matrix H_cpack = mat_transpose(H);
	int one_H_cols = H_cpack.size();
	int one_H_rows = H_cpack[0].size() / batchsize;
	vector<vector<tensor>> expand_H_tensors(one_H_cols, vector<tensor>(one_H_rows));

	const int total_elements = batchsize * one_H_rows;
	#pragma omp parallel for num_threads(32)
	for (int j = 0; j < one_H_cols; ++j) {
		for (int i = 0; i < one_H_rows; ++i) {
			auto& current_tensor = expand_H_tensors[j][i];
			current_tensor.resize(total_elements); 
			int index = 0;
			for (int ell = 0; ell < batchsize; ++ell) {
				const auto& val = H_cpack[j][ell * one_H_rows + i];
				fill_n(current_tensor.begin() + index, one_U_rows, val);
				index += one_U_rows;
			}
		}
	}
	
	// Compute [RW] = [L] * H
	cipherMatrix RW_cpack_ct(one_H_cols);
	
	#pragma omp parallel for num_threads(32)
	for(int j = 0; j < one_H_cols; j++)
	{
		vector<Ciphertext> hat_x_j_ct_items(one_H_rows);
		for(int i = 0; i < one_H_rows; i++)
		{
			Plaintext expand_H_vector_pt;			
			ckks->encoder->encode(expand_H_tensors[j][i], E_cpack[i].parms_id(), E_cpack[i].scale(), expand_H_vector_pt);
			ckks->evaluator->multiply_plain(E_cpack[i], expand_H_vector_pt, hat_x_j_ct_items[i]);
		}
		ckks->evaluator->add_many(hat_x_j_ct_items, RW_cpack_ct[j]);
		ckks->evaluator->rescale_to_next_inplace(RW_cpack_ct[j]);
	}

	double right_time = func_timer.end();
	right_time *= headnumber;
	return make_tuple(RW_cpack_ct, left_time+right_time);
} 

cipherMatrix server_online(CKKSEvaluator* ckks, matrix& hat_X, matrix& weight, cipherMatrix& cipher_RW)
{
	Rectime timer_func;
	timer_func.start();
	matrix hat_XW = mat_mul_mat(hat_X, weight);
	// timer_func.elapsed("--Server compute hat_XW");
	matrix hat_XW_cpack = mat_transpose(hat_XW);

	cipherMatrix cipher_AW(cipher_RW.size());
	#pragma omp parallel for num_threads(32)
	for(int i = 0; i < hat_XW_cpack.size(); i++)
	{
		Plaintext pt;
		ckks->encoder->encode(hat_XW_cpack[i], cipher_RW[i].parms_id(), cipher_RW[i].scale(), pt);
		ckks->evaluator->add_plain(cipher_RW[i], pt, cipher_AW[i]);
	}
	// timer_func.elapsed("--Server compute cipher_AW");
	return cipher_AW;
}

vector<vector<Plaintext>> server_extend_weight(CKKSEvaluator* ckks, const cipherMatrix& cMat_cpack, const matrix& pMat)
{
	int crows = cMat_cpack[0].poly_modulus_degree() / 2; 
	int ccols = cMat_cpack.size();
	int prows = pMat.size();
	int pcols = pMat[0].size();
	vector<vector<Plaintext>> extend_pMat_pts(prows, vector<Plaintext>(pcols));
	if(ccols != prows)
	{
		print_error("cMat_mul_pMat dimensions invalid!");
		return extend_pMat_pts;
	}	
	auto params_id = cMat_cpack[0].parms_id();
	auto scale =  cMat_cpack[0].scale();
	#pragma omp parallel for num_threads(32)
	for (int i = 0; i < prows; ++i) {
        for (int j = 0; j < pcols; ++j) {
            tensor repeated_tensor(crows, pMat[i][j]);
			ckks->encoder->encode(repeated_tensor, params_id, scale, extend_pMat_pts[i][j]);
        }
    }
	return extend_pMat_pts;
}

cipherMatrix cMat_mul_pMat(CKKSEvaluator* ckks, const cipherMatrix& cipher_left_matrix_input_cpack, const matrix& right_matrix_input_cpack)
{
	// Extend to merged replicated vectors
	int one_cols = right_matrix_input_cpack.size();
	int one_rows = right_matrix_input_cpack[0].size();
	vector<vector<Plaintext>> expand_tensors = server_extend_weight(ckks, cipher_left_matrix_input_cpack, mat_transpose(right_matrix_input_cpack));
	cipherMatrix output_cpack_ct(one_cols);
	#pragma omp parallel for num_threads(32)
	for(int j = 0; j < one_cols; j++)
	{
		vector<Ciphertext> ct_items(one_rows);
		for(int i = 0; i < one_rows; i++)
		{
			ckks->evaluator->multiply_plain(cipher_left_matrix_input_cpack[i], expand_tensors[i][j], ct_items[i]);
		}
		ckks->evaluator->add_many(ct_items, output_cpack_ct[j]);
		ckks->evaluator->rescale_to_next_inplace(output_cpack_ct[j]);
	}
	return output_cpack_ct;
}

vector<Ciphertext> extend_batched_ct_rotates(CKKSEvaluator* ckks, const Ciphertext& ct, int rows)
{
	vector<Ciphertext> batched_rotated_cts(rows);
	#pragma omp parallel for num_threads(32)
	for(int i = 0; i < rows; i++)
	{
		tensor mask_left, mask_right;
		tensor single_left(rows-i, 1), single_right(rows-i, 0);
		single_left.insert(single_left.end(), i, 0);
		single_right.insert(single_right.end(), i, 1);
		for(int j = 0; j < batchsize; j++)
		{
			mask_left.insert(mask_left.end(), single_left.begin(), single_left.end());
			mask_right.insert(mask_right.end(), single_right.begin(), single_right.end());
		}
		Plaintext mask_left_pt, mask_right_pt;
		ckks->encoder->encode(mask_left, ct.parms_id(), ct.scale(), mask_left_pt);
		ckks->encoder->encode(mask_right, ct.parms_id(), ct.scale(), mask_right_pt);
		Ciphertext left_rotated_ct, right_rotated_ct;
		ckks->evaluator->rotate_vector(ct, i, *(ckks->galois_keys_allrotate), left_rotated_ct);
		ckks->evaluator->multiply_plain_inplace(left_rotated_ct, mask_left_pt);
		ckks->evaluator->rotate_vector(ct, i-rows, *(ckks->galois_keys_allrotate), right_rotated_ct);
		ckks->evaluator->multiply_plain_inplace(right_rotated_ct, mask_right_pt);

		ckks->evaluator->add(left_rotated_ct, right_rotated_ct, batched_rotated_cts[i]);
		ckks->evaluator->rescale_to_next_inplace(batched_rotated_cts[i]);
	}
	return batched_rotated_cts;
}

cipherMatrix colpack_cMat_mul_cMat(CKKSEvaluator* ckks, const cipherMatrix& left_cpack, const vector<vector<Ciphertext>>& right_cpack_rotates)
{
	cipherMatrix resMat_diagpack;
	int one_rows = input_tokens;
	int one_cols = dim_model_weight_out;
	resMat_diagpack.resize(one_rows);
	cipherMatrix cp_left_cpack = left_cpack;
	
	for(int j = 0; j < one_rows; j++)
	{
		vector<Ciphertext> col_mul_cts(one_cols);
		for(int i = 0; i < one_cols; i++)
		{
			ckks->evaluator->mod_switch_to_inplace(cp_left_cpack[i], right_cpack_rotates[i][j].parms_id());
			ckks->evaluator->multiply(cp_left_cpack[i], right_cpack_rotates[i][j], col_mul_cts[i]);
		}
		ckks->evaluator->add_many(col_mul_cts, resMat_diagpack[j]);
		ckks->evaluator->relinearize_inplace(resMat_diagpack[j], *(ckks->relin_keys));
		ckks->evaluator->rescale_to_next_inplace(resMat_diagpack[j]);
	}
	return resMat_diagpack;
}

cipherMatrix diagpack_cMat_mul_cMat(CKKSEvaluator* ckks, cipherMatrix& left_diagpack, const vector<vector<Ciphertext>>& right_cpack_rotates)
{
	// Rectime func_timer;
	// func_timer.start();
	
	int lrows = left_diagpack[0].poly_modulus_degree() / 2 / batchsize; 
	int lcols = left_diagpack.size();
	int rrows = right_cpack_rotates[0][0].poly_modulus_degree() / 2 / batchsize; 
	int rcols = right_cpack_rotates.size();
	cipherMatrix resMat_cpack(rcols);
	if(lcols != rrows)
	{
		print_error("diagpack_cMat_mul_cMat dimensions invalid!");
		return resMat_cpack;
	}
	
	vector<vector<Ciphertext>> pack_cts(rcols, vector<Ciphertext>(lcols));
	#pragma omp parallel for num_threads(32)
	for(int j = 0; j < lcols; j++)
	{
		for(int i = 0; i < rcols; i++)
		{
			ckks->evaluator->mod_switch_to_inplace(left_diagpack[j], right_cpack_rotates[i][j].parms_id());
			ckks->evaluator->multiply(left_diagpack[j], right_cpack_rotates[i][j], pack_cts[i][j]);
		}
	}
	#pragma omp parallel for num_threads(32)
	for(int i = 0; i < rcols; i++)
	{
		ckks->evaluator->add_many(pack_cts[i], resMat_cpack[i]);
		ckks->evaluator->rescale_to_next_inplace(resMat_cpack[i]);
	}
	// func_timer.elapsed("diagpack_cMat_mul_cMat");
	return resMat_cpack;
}