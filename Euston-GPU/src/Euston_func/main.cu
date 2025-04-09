#include "utils.cuh"
#include "ckks_evaluator.cuh"
#include "myread.cuh"
#include "matrix_mul.cuh"
#include "softmax.cuh"
#include "layernorm.cuh"
#include "gelu.cuh"

vector<string> TEST_TARGETS = {
	"IWMM:Input-Weight MatMul", 
	"CRMM:Colpacked-Rowpacked MatMul", 
	"DCMM:Diagpacked-Colpacked MatMul", 
	"CPMM:Cipher-Plain MatMul",
	"SoftMax",  
	"LayerNorm", 
	"GELU"
};

int main(int argc, char* argv[]) {
	int TEST_TARGET_IDX;
	string TEST_TARGET;
	string model;
	int opt, option_index;

	bool has_b = false, has_m = false, has_p = false;
    while ((opt = getopt_long(argc, argv, "p:b:m:", long_options, &option_index)) != -1) {
        switch (opt) {
			case 'p':
                has_p = true;
				TEST_TARGET_IDX = stoi(optarg);
				TEST_TARGET = TEST_TARGETS[TEST_TARGET_IDX];
				if(TEST_TARGET_IDX != 0)
				{
					has_m = true;
				}
                break;
            case 'b':
				has_b = true;
				batchsize = stoi(optarg);
				input_tokens = 4096 / batchsize;
                break;
            case 'm':
				has_m = true;
                model = string(optarg);
                break;
            case '?':
                std::cerr << "参数错误" << std::endl;
                return 0;
        }
    }
	if (!has_b || !has_m || !has_p) {
        std::cerr << "错误：以下选项为必填项：" << std::endl;
		if (!has_p) std::cerr << "  -p/--protocol" << std::endl;
		if (!has_b) std::cerr << "  -b/--batchsize" << std::endl;
		if (!has_m) std::cerr << "  -m/--model" << std::endl;
    }
	if(model == "BERT")
	{
		dim_model_weight_in = 768;
		dim_model_weight_out = 64;
		headnumber = 12;
	}
	else if(model == "GPT")
	{
		dim_model_weight_in = 1600;
		dim_model_weight_out = 64;
		headnumber = 25;
	}
	else if(model == "LLAMA")
	{
		dim_model_weight_in = 4096;
		dim_model_weight_out = 128;
		headnumber = 32;
	}
	else
	{
		cout << "Model Defination Error." << endl;
		return 0;
	}
	Rectime timer;
	if(TEST_TARGET == TEST_TARGETS[0])
	{
		if(batchsize != 32) std::cerr << " batchsize must be 32 in MatMul." << std::endl;
		vector<int> MATRIX_COEFF_MODULI = {60, 40, 40, 40, 60};
		CKKSEvaluator ckks_client_mat = construct_ckks_evaluator_matrix(MATRIX_COEFF_MODULI);
		CKKSEvaluator ckks = ckks_client_mat;
		matrix hat_input, weight, output;
		
		hat_input = loadtxt(dname+"/input/IWMM_"+model+"_input_32x128X"+to_string(dim_model_weight_in)+".txt");
		weight = loadtxt(dname+"/input/IWMM_"+model+"_weight_"+to_string(dim_model_weight_in)+"x"+to_string(dim_model_weight_out)+".txt");
		output = loadtxt(dname+"/output/IWMM_"+model+"_output_32x128X"+to_string(dim_model_weight_out)+".txt");
		double off_time = 0.0;
		timer.start();
		auto [hat_U_cpack, hat_V, hat_D_ct, hat_R] = client_offline(&ckks);
		double time_off_client = timer.end();
		cout << "Client offline Runtime: " << time_off_client << " ms" << endl;
		off_time += time_off_client;


		auto [cipher_RW,time_off_server] = server_offline(&ckks, hat_U_cpack, hat_V, hat_D_ct, weight);
		cout << "Server offline Runtime: " << time_off_server << " ms" << endl;
		off_time += time_off_server;
		cout << "Offline Total Runtime: " << off_time << " ms" << endl;

		double on_time = 0.0;
		timer.start();
		matrix hat_X = client_online(&ckks, hat_input, hat_R);
		auto time_on_client = timer.end();
		on_time += time_on_client;

		timer.start();
		cipherMatrix cipher_output = server_online(&ckks, hat_X, weight, cipher_RW);
		auto time_on_server = timer.end();
		on_time += time_on_server * headnumber;
		cout << "Online Phase amortized computation runtime: " << on_time / 32 << " ms" << endl;
		ckks.print_chain_index(cipher_output[0], "cipher_output[0]");
	}
	else if(TEST_TARGET == TEST_TARGETS[1])
	{
		if(batchsize != 32) std::cerr << " batchsize must be 32 in MatMul." << std::endl;
		vector<int> MATRIX_COEFF_MODULI = {60, 40, 40, 40, 60};
		CKKSEvaluator ckks_client_mat = construct_ckks_evaluator_matrix(MATRIX_COEFF_MODULI);
		CKKSEvaluator ckks = ckks_client_mat;
		matrix matrix_input = loadtxt(dname+"/input/CCMM_"+model+"_input_32x128X"+to_string(dim_model_weight_out)+".txt");
		matrix matrix_input_cpack = mat_transpose(matrix_input);
		matrix output = loadtxt(dname+"/output/CCMM_"+model+"_output_32x128X"+to_string(dim_model_weight_out)+"_diagpack.txt");

		cipherMatrix cipher_Q_cpack(dim_model_weight_out), cipher_K_cpack(dim_model_weight_out);
		for(int i = 0; i < dim_model_weight_out; i++)
		{
			PhantomPlaintext pt;
			ckks.encoder->encode(matrix_input_cpack[i], ckks.scale, pt);
			ckks.encryptor->encrypt(pt, cipher_Q_cpack[i]);
			ckks.encoder->encode(matrix_input_cpack[i], ckks.scale, pt);
			ckks.encryptor->encrypt(pt, cipher_K_cpack[i]);
		}
		ckks.print_chain_index(cipher_Q_cpack[0], "cipher_Q_cpack[0]");
		vector<vector<PhantomCiphertext>> K_cpack_rotates(matrix_input_cpack.size());
		timer.start();
		for(int i = 0; i < matrix_input_cpack.size(); i++)
		{
			K_cpack_rotates[i] = extend_batched_ct_rotates(&ckks, cipher_K_cpack[i], 128);
		}
		cipherMatrix cipher_QK_diagpack = colpack_cMat_mul_cMat(&ckks, cipher_Q_cpack, K_cpack_rotates);
		auto tim = timer.end();
		cout << "Colpack_cMat_mul_rowpack_cMat 32-batched 128X"+to_string(dim_model_weight_out)+" * "+to_string(dim_model_weight_out)+"X128 Runtime: " << tim << " ms" << endl;
		ckks.print_chain_index(cipher_QK_diagpack[0], "cipher_QK_diagpack[0]");
		double errors = 0;
		output = mat_transpose(output);
		for(int i = 0; i < output.size(); i++)
		{
			errors += ckks.calculate_errors(cipher_QK_diagpack[i], output[i]);
		}
		cout << fixed << setprecision(10) << "Average errors: " << errors / output.size() << endl;
	}
	else if(TEST_TARGET == TEST_TARGETS[2])
	{
		if(batchsize != 32) std::cerr << " batchsize must be 32 in MatMul." << std::endl;
		vector<int> MATRIX_COEFF_MODULI = {60, 40, 40, 40, 40, 40, 60};
		CKKSEvaluator ckks_client_mat = construct_ckks_evaluator_matrix(MATRIX_COEFF_MODULI);
		CKKSEvaluator ckks = ckks_client_mat;
		matrix left_matrix_input_diagpack = loadtxt(dname+"/input/DCMM_"+model+"_left_input_32x128X128.txt");
		left_matrix_input_diagpack = mat_transpose(left_matrix_input_diagpack);
		matrix right_matrix_input = loadtxt(dname+"/input/DCMM_"+model+"_right_input_32x128X"+to_string(dim_model_weight_out)+".txt");
		matrix right_matrix_input_cpack = mat_transpose(right_matrix_input);
		matrix output = loadtxt(dname+"/output/DCMM_"+model+"_output_32x128X"+to_string(dim_model_weight_out)+".txt");

		cipherMatrix cipher_left_matrix_input_diagpack(input_tokens), cipher_right_matrix_input_cpack(dim_model_weight_out);
		for(int i = 0; i < input_tokens; i++)
		{
			PhantomPlaintext pt;
			ckks.encoder->encode(left_matrix_input_diagpack[i], ckks.scale, pt);
			ckks.encryptor->encrypt(pt, cipher_left_matrix_input_diagpack[i]);
		}
		for(int i = 0; i < dim_model_weight_out; i++)
		{
			PhantomPlaintext pt;
			ckks.encoder->encode(right_matrix_input_cpack[i], ckks.scale, pt);
			ckks.encryptor->encrypt(pt, cipher_right_matrix_input_cpack[i]);
		}
		ckks.print_chain_index(cipher_right_matrix_input_cpack[0], "cipher_right_matrix_input_cpack[0]");
		timer.start();
		vector<vector<PhantomCiphertext>> right_matrix_input_cpack_rotates(right_matrix_input_cpack.size());
		for(int i = 0; i < right_matrix_input_cpack.size(); i++)
		{
			right_matrix_input_cpack_rotates[i] = extend_batched_ct_rotates(&ckks, cipher_right_matrix_input_cpack[i], input_tokens);
		}
		cipherMatrix cipher_output_cpack = diagpack_cMat_mul_cMat(&ckks, cipher_left_matrix_input_diagpack, right_matrix_input_cpack_rotates);
		timer.elapsed("Diagpack_cMat_mul_colpack_cMat 32-batched 128X128 * 128X"+to_string(dim_model_weight_out)+" Runtime: ");
		ckks.print_chain_index(cipher_output_cpack[0], "cipher_output_cpack[0]");
		double errors = 0;
		output = mat_transpose(output);
		for(int i = 0; i < output.size(); i++)
		{
			errors += ckks.calculate_errors(cipher_output_cpack[i], output[i]);
		}
		cout << "Average errors: " << errors / output.size() << endl;
	}
	else if(TEST_TARGET == TEST_TARGETS[3])
	{
		if(batchsize != 32) std::cerr << " batchsize must be 32 in MatMul." << std::endl;
		int GELU_size;
		if(model == "BERT")
		{
			GELU_size = 3072; //3072=64*48 
		}
		else if(model == "GPT")
		{
			GELU_size = 6400; //6400=64*100
		}
		else if(model == "LLAMA")
		{
			GELU_size = 14336; //14336=64*224
		}
		vector<int> MATRIX_COEFF_MODULI = {60, 40, 40, 60};
		CKKSEvaluator ckks_client_mat = construct_ckks_evaluator_matrix(MATRIX_COEFF_MODULI);
		CKKSEvaluator ckks = ckks_client_mat;
		matrix left_matrix_input,left_matrix_input_cpack,right_matrix_input,right_matrix_input_cpack,output;
		cipherMatrix cipher_output_cpack;
		left_matrix_input = loadtxt(dname+"/input/CPMM1_"+model+"_left_input_32x128X"+to_string(dim_model_weight_in)+".txt");
		left_matrix_input_cpack = mat_transpose(left_matrix_input);
		right_matrix_input = loadtxt(dname+"/input/CPMM1_"+model+"_right_input_"+to_string(dim_model_weight_in)+"X64.txt");
		right_matrix_input_cpack = mat_transpose(right_matrix_input);
		output = loadtxt(dname+"/output/CPMM1_"+model+"_output_32x128X64.txt");

		cipherMatrix cipher_left_matrix_input_cpack(left_matrix_input_cpack.size());
		for(int i = 0; i < left_matrix_input_cpack.size(); i++)
		{
			PhantomPlaintext pt;
			ckks.encoder->encode(left_matrix_input_cpack[i], ckks.scale, pt);
			ckks.encryptor->encrypt(pt, cipher_left_matrix_input_cpack[i]);
		}
		ckks.print_chain_index(cipher_left_matrix_input_cpack[0], "cipher_left_matrix_input_cpack[0]");
		Rectime timer;
		timer.start();
		cipher_output_cpack = cMat_mul_pMat(&ckks, cipher_left_matrix_input_cpack, right_matrix_input_cpack);
		timer.elapsed("cMat_mul_pMat 1 32-batched 128X"+to_string(dim_model_weight_in)+" * "+to_string(dim_model_weight_in)+"X64 Runtime: ");
		ckks.print_chain_index(cipher_output_cpack[0], "cipher_output_cpack[0]");
		double errors = 0;
		matrix output_cpack = mat_transpose(output);
		for(int i = 0; i < output_cpack.size(); i++)
		{
			errors += ckks.calculate_errors(cipher_output_cpack[i], output_cpack[i]);
		}
		cout << "Average errors: " << errors / output_cpack.size() << endl;


		left_matrix_input = loadtxt(dname+"/input/CPMM2_"+model+"_left_input_32x128X"+to_string(GELU_size)+".txt");
		left_matrix_input_cpack = mat_transpose(left_matrix_input);
		right_matrix_input = loadtxt(dname+"/input/CPMM2_"+model+"_right_input_"+to_string(GELU_size)+"X64.txt");
		right_matrix_input_cpack = mat_transpose(right_matrix_input);
		output = loadtxt(dname+"/output/CPMM2_"+model+"_output_32x128X64.txt");

		cipherMatrix cipher_left_matrix_input_cpack2(left_matrix_input_cpack.size());
		for(int i = 0; i < left_matrix_input_cpack.size(); i++)
		{
			PhantomPlaintext pt;
			ckks.encoder->encode(left_matrix_input_cpack[i], ckks.scale, pt);
			ckks.encryptor->encrypt(pt, cipher_left_matrix_input_cpack2[i]);
		}
		ckks.print_chain_index(cipher_left_matrix_input_cpack2[0], "cipher_left_matrix_input_cpack2[0]");
		timer.start();
		cipher_output_cpack = cMat_mul_pMat(&ckks, cipher_left_matrix_input_cpack2, right_matrix_input_cpack);
		timer.elapsed("cMat_mul_pMat 2 32-batched 128X"+to_string(GELU_size)+" * "+to_string(GELU_size)+"X"+to_string(dim_model_weight_out)+" Runtime: "); 
		ckks.print_chain_index(cipher_output_cpack[0], "cipher_output_cpack[0]");
		errors = 0;
		output_cpack = mat_transpose(output);
		for(int i = 0; i < output_cpack.size(); i++)
		{
			errors += ckks.calculate_errors(cipher_output_cpack[i], output_cpack[i]);
		}
		cout << "Average errors: " << errors / output_cpack.size() << endl;
	}
	else if(TEST_TARGET == TEST_TARGETS[4])
	{
		long logN = 8+log2(batchsize);
		vector<int> COEFF_MODULI = {58, 40, 41, 41, 41, 41, 41, 42, 39, 39, 39, 58};
		CKKSEvaluator ckks_server_nonlinear = construct_ckks_evaluator_nonlinear(logN,COEFF_MODULI);
		CKKSEvaluator ckks = ckks_server_nonlinear;
		matrix matrix_input = loadtxt(dname+"/input/Softmax_input_batched_32768X128.txt", pow(2,logN-1));
		matrix input_diagpack = diagpack_matrix(matrix_input);
		matrix softmax_output = loadtxt(dname+"/output/Softmax_output_batched_32768X128.txt", pow(2,logN-1));
		cipherMatrix cipher_input_diagpack(128);
		for(int i = 0; i < 128; i++)
		{
			PhantomPlaintext pt;
			ckks.encoder->encode(input_diagpack[i], pow(2,38), pt);
			ckks.encryptor->encrypt(pt, cipher_input_diagpack[i]);
		}
		ckks.print_chain_index(cipher_input_diagpack[0], "cipher_input_diagpack[0]");
		
		timer.start();
		cipherMatrix cipher_softmax_output = cipher_softmax(&ckks, cipher_input_diagpack);
		double tim = timer.end();
		cout << to_string(batchsize)+"-Batched 128 x 128 cipher softmax amortized computation: " << tim / batchsize << " ms" << endl;
		ckks.print_chain_index(cipher_softmax_output[0], "cipher_softmax_output[0]");
		
		matrix dec_softmax_output_cpack(cipher_softmax_output.size());
		for(int i = 0; i < cipher_softmax_output.size(); i++)
		{
			PhantomPlaintext pt;
			ckks.decryptor->decrypt(cipher_softmax_output[i], pt);
			ckks.encoder->decode(pt, dec_softmax_output_cpack[i]);
		}
		matrix dec_softmax_output = mat_transpose(dec_softmax_output_cpack);
		double errors = 0;
		for(int i = 0; i < dec_softmax_output[0].size(); i++)
		{
			errors += fabs(dec_softmax_output[0][i] - softmax_output[0][i]);
		}
		cout << fixed << setprecision(10) << "Average errors: " << errors / dec_softmax_output[0].size() << endl;
	}
	else if(TEST_TARGET == TEST_TARGETS[5])
	{
		long logN = 8+log2(batchsize);
		vector<int> COEFF_MODULI = {58, 42, 42, 42, 42, 42, 42, 53, 53, 40, 58};
		CKKSEvaluator ckks_server_nonlinear = construct_ckks_evaluator_nonlinear(logN,COEFF_MODULI);
		CKKSEvaluator ckks = ckks_server_nonlinear;
		matrix matrix_input = loadtxt(dname+"/input/LayerNorm_input_batched_32768X768.txt", pow(2,logN-1));
		matrix LN_input_cpack = mat_transpose(matrix_input);
		matrix LN_output = loadtxt(dname+"/output/LayerNorm_output_batched_32768X768.txt", pow(2,logN-1));
		cipherMatrix cipher_LN_input_cpack(768);
		
		for(int i = 0; i < 768; i++)
		{
			PhantomPlaintext pt;
			ckks.encoder->encode(LN_input_cpack[i], pow(2,30.415037499278), pt);
			ckks.encryptor->encrypt(pt, cipher_LN_input_cpack[i]);
		}
		ckks.print_chain_index(cipher_LN_input_cpack[0], "cipher_LN_input_cpack[0]");
		timer.start();
		cipherMatrix cipher_LN_output_cpack = cipher_layernorm(&ckks, cipher_LN_input_cpack);
		auto tim = timer.end();
		cout << to_string(batchsize)+"-Batched 128 x 768 cipher LayerNorm amortized computation: " << tim / batchsize << " ms" << endl;
		ckks.print_chain_index(cipher_LN_output_cpack[0], "cipher_LN_output_cpack[0]");
		
		matrix dec_LN_output_cpack(cipher_LN_output_cpack.size());
		for(int i = 0; i < cipher_LN_output_cpack.size(); i++)
		{
			PhantomPlaintext pt;
			ckks.decryptor->decrypt(cipher_LN_output_cpack[i], pt);
			ckks.encoder->decode(pt, dec_LN_output_cpack[i]);
		}
		matrix dec_LN_output = mat_transpose(dec_LN_output_cpack);
		double errors = 0;
		for(int i = 0; i < dec_LN_output[0].size(); i++)
		{
			errors += fabs(dec_LN_output[0][i] - LN_output[0][i]);
		}
		cout << fixed << setprecision(10) << "Average errors: " << errors / dec_LN_output[0].size() << endl;
	}
	else if(TEST_TARGET == TEST_TARGETS[6])
	{
		long logN = 8+log2(batchsize);
		timer.start();
		vector<int> COEFF_MODULI = {58, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 40, 40, 58};
		vector<int> COEFF_MODULI_SIGN = {58, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 58};
		CKKSEvaluator ckks_server_nonlinear = construct_ckks_evaluator_nonlinear(logN,COEFF_MODULI);
		CKKSEvaluator ckks_sign = construct_ckks_evaluator_nonlinear(logN, COEFF_MODULI_SIGN);
		CKKSEvaluator ckks = ckks_server_nonlinear;
		timer.elapsed("生成加密工具耗时：");
		matrix gelu_input = loadtxt(dname+"/input/gelu_input_batched_32768X3072.txt", pow(2,logN-1));
		matrix gelu_output = loadtxt(dname+"/output/gelu_output_batched_32768X3072.txt", pow(2,logN-1));

		matrix gelu_input_cpack = mat_transpose(gelu_input);
		int col_siz = 3072;
		cipherMatrix cipher_gelu_input_cpack(col_siz), cipher_gelu_input_cpack_signs(col_siz);
		timer.start();
		
		for(int i = 0; i < col_siz; i++)
		{
			PhantomPlaintext pt;
			ckks.encoder->encode(gelu_input_cpack[i], pow(2,40), pt);
			ckks.encryptor->encrypt(pt, cipher_gelu_input_cpack[i]);
		}
		
		timer.elapsed("加密耗时：");
		ckks.print_chain_index(cipher_gelu_input_cpack[0], "cipher_gelu_input_cpack[0]");
		timer.start();
		
		for(int i = 0; i < col_siz; i++)
		{
			PhantomPlaintext pt;
			PhantomCiphertext ct;
			ckks_sign.encoder->encode(gelu_input_cpack[i], pow(2,40), pt);
			ckks_sign.encryptor->encrypt(pt, ct);
			ckks_sign.evaluator->multiply_const_inplace(ct, 1.0 / 6);
			ckks_sign.evaluator->rescale_to_next_inplace(ct);
			PhantomCiphertext cipher_signs = cipher_sign(&ckks_sign, ct, 3, 3, 1); 
			switch_ckks_evaluator(cipher_signs, &ckks_sign, &ckks);
			cipher_gelu_input_cpack_signs[i] = cipher_signs;
		}
		cipherMatrix cipher_gelu_input_cpack_abs(col_siz);
		
		for(int i = 0; i < col_siz; i++)
		{
			ckks.evaluator->multiply(cipher_gelu_input_cpack[i], cipher_gelu_input_cpack_signs[i], cipher_gelu_input_cpack_abs[i]);
			ckks.evaluator->relinearize_inplace(cipher_gelu_input_cpack_abs[i], *(ckks.relin_keys));
			ckks.evaluator->rescale_to_next_inplace(cipher_gelu_input_cpack_abs[i]);
		}

		cipherMatrix gelu = cipher_gelu(&ckks, cipher_gelu_input_cpack_abs);
		cipherMatrix gelu_rectify = cipher_gelu_rectify(&ckks, cipher_gelu_input_cpack_abs, cipher_gelu_input_cpack_signs);
		cipherMatrix cipher_gelu_output_cpack(col_siz);
		
		for(int i = 0; i < col_siz; i++)
		{
			ckks.evaluator->mod_switch_to_inplace(gelu_rectify[i], gelu[i].params_id());
			ckks.evaluator->multiply(gelu[i], gelu_rectify[i], cipher_gelu_output_cpack[i]);
			ckks.evaluator->relinearize_inplace(cipher_gelu_output_cpack[i], *(ckks.relin_keys));
			ckks.evaluator->rescale_to_next_inplace(cipher_gelu_output_cpack[i]);
		}
		auto tim = timer.end();
		cout << to_string(batchsize)+"-Batched 128 x 3072 cipher GELU amortized computation: " << tim  / batchsize << " ms" << endl;
		ckks.print_chain_index(cipher_gelu_output_cpack[1], "cipher_gelu_output_cpack[0]");

		matrix dec_gelu_output_cpack(cipher_gelu_output_cpack.size());
		for(int i = 0; i < cipher_gelu_output_cpack.size(); i++)
		{
			PhantomPlaintext pt;
			ckks.decryptor->decrypt(cipher_gelu_output_cpack[i], pt);
			ckks.encoder->decode(pt, dec_gelu_output_cpack[i]);
		}
		matrix dec_gelu_output = mat_transpose(dec_gelu_output_cpack);
		double errors = 0;
		for(int i = 0; i < dec_gelu_output[0].size(); i++)
		{
			errors += fabs(dec_gelu_output[0][i] - gelu_output[0][i]);
		}
		cout << fixed << setprecision(10) << "Average errors: " << errors / dec_gelu_output[0].size() << endl;
	}
	return 0;
}