#include "matrix_mul.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <typeindex>
#include <filesystem>
#include "seal/util/polyarithsmallmod.h"

using namespace std;
using namespace seal;
using namespace chrono;
using namespace seal::util;
namespace fs = filesystem;

matrix MMEvaluator::transposeMatrix(const matrix &mat) {
  if (mat.empty()) {
    return {};
  }
  int rows = mat.size();
  int cols = mat[0].size();
  matrix transposedMatrix(cols, vector<double>(rows));

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      transposedMatrix[j][i] = mat[i][j];
    }
  }

  return transposedMatrix;
}




matrix MMEvaluator::readMatrix(const string &filename, int rows, int cols) {
  matrix mat(rows, vector<double>(cols));
  ifstream file(filename);

  if (!file.is_open()) {
    cerr << "Can not open file: " << filename << endl;
    return mat;
  }

  string line;
  for (int i = 0; i < rows; ++i) {
    if (getline(file, line)) {
      istringstream iss(line);
      for (int j = 0; j < cols; ++j) {
        if (!(iss >> mat[i][j])) {
          cerr << "read error: " << filename << " (row: " << i << ", column: " << j << ")" << endl;
        }
      }
    }
  }

  file.close();
  return mat;
}
//这是个把一行打包向量加密的算法，具体做法是先把数据放到ckks的模上，然后用加密的0向量加上这个明文得到加密结果
void MMEvaluator::enc_compress_ciphertext(vector<double> &val, Ciphertext &ct) {
  //构造一个全0的加密向量，加密是随机加密的，即每一次对同一个数加密的结果都是不一样的
  Plaintext zero_pt;
  ckks->encoder->encode(vector<double>(ckks->N / 2, 0.0), ckks->scale, zero_pt);
	// cout << zero_pt.nonzero_coeff_count() << endl;
	// cout << zero_pt.coeff_count() << endl;
  Ciphertext zero;
  ckks->encryptor->encrypt(zero_pt, zero); 


  auto context = *ckks->context;
  auto context_data = context.get_context_data(context.first_parms_id());
  auto param = context_data->parms();
	auto &coeff_modulus = param.coeff_modulus();
	std::size_t coeff_modulus_size = coeff_modulus.size();
  auto ntt_tables = context_data->small_ntt_tables();
	
  auto poly_modulus_degree = ckks->degree;
	// cout << ckks-> N << "    " << poly_modulus_degree << endl;
  Plaintext p(poly_modulus_degree * coeff_modulus_size);
	p.set_zero();
	// cout << p.coeff_count() << endl;
  //这一段，把val里面的每个数放大，然后分别对param的两个模数取模，分别放到p的两个poly_modulus_degree块。正负号都会保留住。
  for (auto i = 0; i < val.size(); i++) {
    auto coeffd = round(val[i] * 10000000000);
    bool is_negative = signbit(coeffd);
    auto coeffu = static_cast<uint64_t>(fabs(coeffd));
    if (is_negative) { 
      for (size_t j = 0; j < coeff_modulus_size; j++) {
        p[i + (j * poly_modulus_degree)] = util::negate_uint_mod(
            util::barrett_reduce_64(coeffu, param.coeff_modulus()[j]), param.coeff_modulus()[j]);
      }
    } else {
      for (size_t j = 0; j < coeff_modulus_size; j++) {
        p[i + (j * poly_modulus_degree)] = util::barrett_reduce_64(coeffu, param.coeff_modulus()[j]);
      }
    }
  }
	
  for (size_t i = 0; i < coeff_modulus_size; i++) {
		// cout << p.to_string() << endl;
    util::ntt_negacyclic_harvey(p.data(i * poly_modulus_degree), ntt_tables[i]);
	 	util::inverse_ntt_negacyclic_harvey(p.data(i * poly_modulus_degree), ntt_tables[i]);
		// cout << p.to_string() << endl;
		util::ntt_negacyclic_harvey(p.data(i * poly_modulus_degree), ntt_tables[i]);
  }
  p.parms_id() = context.first_parms_id();
  p.scale() = 10000000000;
	// ckks->print_decoded_pt(p, 8192);
  zero.scale() = p.scale();

  ckks->evaluator->add_plain(zero, p, ct);

}

//这个就是NEXUS提出的密文解包算法
vector<Ciphertext> MMEvaluator::expand_ciphertext(const Ciphertext &encrypted, uint32_t m, GaloisKeys &galkey, vector<uint32_t> &galois_elts) {
  uint32_t logm = ceil(log2(m));
  auto n = ckks->N;

  vector<Ciphertext> temp;
  temp.push_back(encrypted);
  
  Ciphertext tempctxt_rotated;
  Ciphertext tempctxt_shifted;
  Ciphertext tempctxt_rotatedshifted;

  for (uint32_t i = 0; i < logm; i++) {
    vector<Ciphertext> newtemp(temp.size() << 1);
    int index_raw = (n << 1) - (1 << i); //2n-2^i
	// cout << "index_raw: " << "2n-2^" << i << endl;
    int index = (index_raw * galois_elts[i]) % (n << 1); // (2n-2^i)*(n/2^i + 1) mod 2n
	// cout << "index: " <<  ckks->rots[i] << endl;

    for (uint32_t a = 0; a < temp.size(); a++) {
      ckks->evaluator->apply_galois(temp[a], ckks->rots[i], *(ckks->galois_keys), tempctxt_rotated);  // sub
      ckks->evaluator->add(temp[a], tempctxt_rotated, newtemp[a]);
      multiply_power_of_x(temp[a], tempctxt_shifted, index_raw);  // x**-1
      // if (temp.size() == 1) ckks->print_decrypted_ct(tempctxt_shifted, 10);
      multiply_power_of_x(tempctxt_rotated, tempctxt_rotatedshifted, index);
      // if (temp.size() == 1) ckks->print_decrypted_ct(tempctxt_rotatedshifted, 10);
      ckks->evaluator->add(tempctxt_shifted, tempctxt_rotatedshifted, newtemp[a + temp.size()]);
    }
    temp = newtemp;
  }
  return temp;
}

void MMEvaluator::multiply_power_of_x(Ciphertext &encrypted, Ciphertext &destination, int index) {
  // Method 1:
  // string s = "";
  // destination = encrypted;
  // while (index >= ckks->N - 1) {
  //     s = "1x^" + to_string(ckks->N - 1);
  //     Plaintext p(s);
  //     ckks->evaluator->multiply_plain(destination, p, destination);
  //     index -= ckks->N - 1;
  // }

  // s = "1x^" + to_string(index);

  // Plaintext p(s);
  // ckks->evaluator->multiply_plain(destination, p, destination);

  // Method 2:
  auto context = *ckks->context;
  auto context_data = context.get_context_data(context.first_parms_id());
  auto param = context_data->parms();

  ckks->evaluator->transform_from_ntt_inplace(encrypted);
  auto coeff_mod_count = param.coeff_modulus().size();
  auto coeff_count = ckks->degree;
  auto encrypted_count = encrypted.size();

  destination = encrypted;

  for (int i = 0; i < encrypted_count; i++) {
    for (int j = 0; j < coeff_mod_count; j++) {
      negacyclic_shift_poly_coeffmod(
          encrypted.data(i) + (j * coeff_count),
          coeff_count,
          index,
          param.coeff_modulus()[j],
          destination.data(i) + (j * coeff_count));
    }
  }

  ckks->evaluator->transform_to_ntt_inplace(encrypted);
  ckks->evaluator->transform_to_ntt_inplace(destination);
}


double cal_ct_size(Ciphertext ct)
{
	vector<seal::seal_byte> rece_bytes(ct.save_size());
	size_t siz = ct.save(rece_bytes.data(), rece_bytes.size());
	double total_mb = static_cast<double>(siz) / 1024.0 / 1024.0;
	return total_mb;
}

double cal_pt_size(matrix pt)
{
	size_t total_bytes = 0;
    for (const auto& inner_vec : pt) {
        total_bytes += inner_vec.size() * sizeof(double);
    }

	// 转换为 MB
	double total_mb = static_cast<double>(total_bytes) / 1024.0 / 1024.0;
	return total_mb;
}

void MMEvaluator::matrix_mul(matrix &x, matrix &matrix_weight_T, vector<Ciphertext> &res) {
	matrix y;
	double offline_time = 0.0;
	Rectime timer;
	timer.start();
	vector<double> row_ct;
	for (auto i = 0; i < dim_model_weight_in * dim_model_weight_out; i++) {
		int row = i / dim_model_weight_in; 
		int col = i % dim_model_weight_in;
		row_ct.push_back(matrix_weight_T[row][col]);
		if (i % ckks->degree == (ckks->degree - 1)) {
			y.push_back(row_ct);
			row_ct.clear();
		}
	}
	if(!row_ct.empty())
		y.push_back(row_ct);

	vector<Ciphertext> b_compressed_cts;
	for (auto &yy: y) {
		Ciphertext ct;
		enc_compress_ciphertext(yy, ct);
		b_compressed_cts.push_back(ct);
	}

	auto time_off_server = timer.end();
	offline_time += time_off_server * headnumber;

	double offline_commun = 0.0;
	double offline_server_commun = 0.0;
	for (auto &ct : b_compressed_cts) {
		offline_server_commun += cal_ct_size(ct);
	}
	offline_server_commun *= headnumber;
	cout << "Client offline Storage: " << offline_server_commun+cal_pt_size(x) << " MB" << endl;
	offline_commun += offline_server_commun;
	

	timer.start();
	auto st = chrono::high_resolution_clock::now();
	vector<Ciphertext> b_expanded_cts;
	for (auto i = 0; i < b_compressed_cts.size(); i++) {
		vector<Ciphertext> temp_cts = expand_ciphertext(b_compressed_cts[i], ckks->degree, *ckks->galois_keys, ckks->rots);
		cout << "Expanded ciphertext #" << i + 1 << endl;
		b_expanded_cts.insert(b_expanded_cts.end(), make_move_iterator(temp_cts.begin()), make_move_iterator(temp_cts.end()));
	}
	auto en = chrono::high_resolution_clock::now();
	cout << "Expand ciphertext once a time: " << chrono::duration_cast<chrono::milliseconds>(en - st).count() / b_compressed_cts.size()<< " ms" << endl;
	vector<Plaintext> a_pts;
	a_pts.reserve(dim_model_weight_in);

	for (int i = 0; i < dim_model_weight_in; i++) {
		Plaintext pt;
		ckks->encoder->encode(x[i], ckks->scale, pt);
		a_pts.emplace_back(pt);
	}

	Ciphertext temp;
	for (int i = 0; i < dim_model_weight_out; i++) {
		Ciphertext res_col_ct;
		vector<Ciphertext> temp_cts(dim_model_weight_in);

		for (int j = 0; j < dim_model_weight_in; j++) {
			ckks->evaluator->multiply_plain(b_expanded_cts[i * dim_model_weight_in + j], a_pts[j], temp_cts[j]);
		}

		res_col_ct.scale() = temp_cts[0].scale();
		ckks->evaluator->add_many(temp_cts, res_col_ct);

		res_col_ct.scale() *= ckks->scale;
		res.push_back(res_col_ct);
	}

	for (auto &ct : res) {
		while (ct.coeff_modulus_size() > 1) {
		ckks->evaluator->rescale_to_next_inplace(ct);
		}
	}

	auto time_off_client = timer.end();
	time_off_client *= headnumber;
	cout << "Client offline Runtime: " << time_off_client << " ms" << endl;
	offline_time += time_off_client;
	cout << "Offline Total Runtime: " << offline_time << " ms" << endl;
	double offline_client_commun = 0.0;
	for (auto &ct : res) {
		offline_client_commun += cal_ct_size(ct);
	}
	offline_client_commun *= headnumber;
	offline_commun += offline_client_commun;
	cout << "Client offline Communication: " << offline_commun << " MB" << endl;
}
