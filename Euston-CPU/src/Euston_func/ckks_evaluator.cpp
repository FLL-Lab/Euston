#include "ckks_evaluator.h"


using namespace std;
using namespace seal;
using namespace chrono;
using namespace seal::util;

double SCALE = pow(2.0, 40);

void CKKSEvaluator::print_chain_index(Ciphertext ct, string ct_name)
{
	cout << fixed << setprecision(10);
	cout << "    + Modulus chain index for " + ct_name << " : "
         << context->get_context_data(ct.parms_id())->chain_index() << endl;
	cout << "    + Scale of " + ct_name << " : "  << log2(ct.scale()) << " bits" << endl;
}

void CKKSEvaluator::print_dec(Ciphertext ct, string des, int num)
{
	cout << fixed << setprecision(6);
	Plaintext pt;
	decryptor->decrypt(ct, pt);
	tensor msg;
	encoder->decode(pt, msg);
	cout << des;
	for(int i = 0; i < num; i++)
	{
		cout << " " << msg[i];
	}
	cout << endl;
}
double CKKSEvaluator::calculate_errors(Ciphertext ct, tensor list)
{
	Plaintext pt;
	tensor dec;
	decryptor->decrypt(ct, pt);
	encoder->decode(pt, dec);
	double errors;
	if(list.size() != dec.size())
	{
		cout << dec.size() << endl;
		cout << list.size() << endl;
		print_error("Dimensions invalid when calculate errors.");
	}
	double maxxx = -1;
	for(int i = 0; i < list.size(); i++)
	{
		if(fabs(list[i]-dec[i]) > maxxx)
		{
			maxxx = fabs(list[i]-dec[i]);
		}
		errors += fabs(list[i]-dec[i]);
	}
	// cout << fixed << setprecision(5) << maxxx << endl;
	// cout << fixed << setprecision(5) << errors << endl;
	// cout << endl;
	return errors / list.size();
}

CKKSEvaluator construct_ckks_evaluator_matrix(vector<int> coeffs)
{
    long logN = 13;
    size_t poly_modulus_degree = 1 << logN;

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, coeffs));

    auto context = make_shared<SEALContext>(parms, true, sec_level_type::none);

    auto keygen = make_shared<KeyGenerator>(*context);
    auto secret_key = keygen->secret_key();
    auto public_key = make_shared<PublicKey>();
    keygen->create_public_key(*public_key);
    auto relin_keys = make_shared<RelinKeys>();
    keygen->create_relin_keys(*relin_keys);
    auto galois_keys = make_shared<GaloisKeys>();
    auto galois_keys_allrotate = make_shared<GaloisKeys>();

    vector<uint32_t> rots;
    for (int i = 0; i < logN; i++) {
        rots.push_back((poly_modulus_degree + exponentiate_uint(2, i)) / exponentiate_uint(2, i));
    }
    keygen->create_galois_keys(rots, *galois_keys);
    keygen->create_galois_keys(*galois_keys_allrotate);

    auto encryptor = make_shared<Encryptor>(*context, *public_key);
    auto encoder = make_shared<CKKSEncoder>(*context);
    auto evaluator = make_shared<Evaluator>(*context, *encoder);
    auto decryptor = make_shared<Decryptor>(*context, secret_key);

    CKKSEvaluator ckks_evaluator(context, encryptor, decryptor, encoder, evaluator, relin_keys, galois_keys, galois_keys_allrotate, SCALE);
    return ckks_evaluator;
}

CKKSEvaluator construct_ckks_evaluator_nonlinear(long logN, vector<int> coeffs)
{
    size_t poly_modulus_degree = 1 << logN;

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, coeffs));

    auto context = make_shared<SEALContext>(parms, true, sec_level_type::none);

    auto keygen = make_shared<KeyGenerator>(*context);
    auto secret_key = keygen->secret_key();
    auto public_key = make_shared<PublicKey>();
    keygen->create_public_key(*public_key);
    auto relin_keys = make_shared<RelinKeys>();
    keygen->create_relin_keys(*relin_keys);
    auto galois_keys = make_shared<GaloisKeys>();
    auto galois_keys_allrotate = make_shared<GaloisKeys>();
    keygen->create_galois_keys(*galois_keys);
    keygen->create_galois_keys(*galois_keys_allrotate);

    auto encryptor = make_shared<Encryptor>(*context, *public_key);
    auto encoder = make_shared<CKKSEncoder>(*context);
    auto evaluator = make_shared<Evaluator>(*context, *encoder);
    auto decryptor = make_shared<Decryptor>(*context, secret_key);

    CKKSEvaluator ckks_evaluator(context, encryptor, decryptor, encoder, evaluator, relin_keys, galois_keys, galois_keys_allrotate, SCALE);
    return ckks_evaluator;
}

void switch_ckks_evaluator(Ciphertext& ct, CKKSEvaluator* ckks1, CKKSEvaluator* ckks2)
{
	Plaintext pt;
	tensor ten;
	ckks1->decryptor->decrypt(ct, pt);
	ckks1->encoder->decode(pt, ten);
	ckks2->encoder->encode(ten, ckks2->scale, pt);
	ckks2->encryptor->encrypt(pt, ct);
}

void encrypting_compress(const CKKSEvaluator* ckks, const tensor& list, Ciphertext &ct)
{
	Plaintext zero_pt;
	ckks->encoder->encode(vector<double>(ckks->N / 2, 0.0), ckks->scale, zero_pt);
	Ciphertext zero;
	ckks->encryptor->encrypt(zero_pt, zero); 


	auto context = *ckks->context;
	auto context_data = context.get_context_data(context.first_parms_id());
	auto param = context_data->parms();
	auto &coeff_modulus = param.coeff_modulus();
	size_t coeff_modulus_size = coeff_modulus.size();
	auto ntt_tables = context_data->small_ntt_tables();
		
	auto poly_modulus_degree = ckks->degree;
	Plaintext p(poly_modulus_degree * coeff_modulus_size);
	p.set_zero();
	for (auto i = 0; i < list.size(); i++) {
		auto coeffd = round(list[i] * (SCALE / ckks->N));
		bool is_negative = signbit(coeffd);
		auto coeffu = static_cast<uint64_t>(fabs(coeffd));
		if (is_negative) { 
		for (size_t j = 0; j < coeff_modulus_size; j++) {
			p[i + (j * poly_modulus_degree)] = negate_uint_mod(
				barrett_reduce_64(coeffu, param.coeff_modulus()[j]), param.coeff_modulus()[j]);
		}
		} else {
		for (size_t j = 0; j < coeff_modulus_size; j++) {
			p[i + (j * poly_modulus_degree)] = barrett_reduce_64(coeffu, param.coeff_modulus()[j]);
		}
		}
	}
		
	for (size_t i = 0; i < coeff_modulus_size; i++) {
		ntt_negacyclic_harvey(p.data(i * poly_modulus_degree), ntt_tables[i]);
	}
	p.parms_id() = context.first_parms_id();
	p.scale() = (SCALE / ckks->N);
	zero.scale() = p.scale();

	ckks->evaluator->add_plain(zero, p, ct);
}

vector<Ciphertext> encrypting_decompress(const CKKSEvaluator* ckks, const Ciphertext &encrypted, uint32_t m, GaloisKeys &galkey, const vector<uint32_t> &galois_elts) 
{
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
		int index = (index_raw * galois_elts[i]) % (n << 1); // (2n-2^i)*(n/2^i + 1) mod 2n

		for (uint32_t a = 0; a < temp.size(); a++) 
		{
			ckks->evaluator->apply_galois(temp[a], ckks->rots[i], *(ckks->galois_keys), tempctxt_rotated);  // sub
			ckks->evaluator->add(temp[a], tempctxt_rotated, newtemp[a]);
			multiply_power_of_x(ckks, temp[a], tempctxt_shifted, index_raw);  // x**-1
			// if (temp.size() == 1) ckks->print_decrypted_ct(tempctxt_shifted, 10);
			multiply_power_of_x(ckks, tempctxt_rotated, tempctxt_rotatedshifted, index);
			// if (temp.size() == 1) ckks->print_decrypted_ct(tempctxt_rotatedshifted, 10);
			ckks->evaluator->add(tempctxt_shifted, tempctxt_rotatedshifted, newtemp[a + temp.size()]);
		}
		temp = newtemp;
	}
	return temp;
}

void multiply_power_of_x(const CKKSEvaluator* ckks, Ciphertext &encrypted, Ciphertext &destination, int index) 
{
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


Ciphertext cipher_exp(CKKSEvaluator* ckks, Ciphertext& x)  // x in [-pow(2,r)*target, 0] 消耗r+3层
{
	Rectime func_timer;
	func_timer.start();
	// ckks->print_chain_index(x, "x");
	// ckks->print_dec(x, "x", 10);
	int r = 3;
	// ckks->evaluator->multiply_const_inplace(x, 1.0 / pow(2,r));
	// ckks->evaluator->rescale_to_next_inplace(x);
	x.scale() *= pow(2,r);
	int degree = 8;
	// ckks->print_dec(x, "x", 1);
	vector<double> a{0.999999807711439, 0.9999928438309771, 0.4999357692100727, 0.166424156797964, 0.041193827292401716, 0.00781230587260862, 0.0010544638554755939, 7.517160876064213e-05};
					//x/(2^r) in [-5,0] approx: 0.9999176045120705, 0.9987416303869157, 0.4953285  812872865, 0.159281595830637, 0.03551592614263832, 0.00533666218566905, 0.00048218432821377466, 1.9546405829954175e-05
					//x/(2^r) in [-4,0] approx: 0.9999791667046215, 0.9996057712501535, 0.49819170081449066, 0.16314976088485714, 0.038088652098859546, 0.006229550870180332, 0.0006372698915844369, 3.018992965556069e-05
					//x/(2^r) in [-3,0] approx: 0.9999968153132669, 0.9999203314486449, 0.4995181577601474, 0.1654360877641768, 0.04003357168254322, 0.007095152387352851, 0.000830588124366306, 4.729606626830923e-05
					//x/(2^r) in [-2,0] approx: 0.999999807711439, 0.9999928438309771, 0.4999357692100727, 0.166424156797964, 0.041193827292401716, 0.00781230587260862, 0.0010544638554755939, 7.517160876064213e-05
	/*
	x0 = a0
	x1 = a1 * x
	x2 = a2 * x^2
	x3 = a3*x * x^2
	x4 = a4 * x^4
	x5 = a5*x * x^4
	x6 = a6*x^2 * x^4
	x7 = a7*x * x^2 * x^4
	*/
	auto relin_keys = *(ckks->relin_keys);
	vector<Ciphertext> xs(degree);
	xs[1] = x;
	Plaintext pt;
	for(int i = 1; i <= 2; i++)
	{
		ckks->evaluator->square(xs[pow(2, i-1)], xs[pow(2, i)]);
		ckks->evaluator->relinearize_inplace(xs[pow(2, i)], relin_keys);
		ckks->evaluator->rescale_to_next_inplace(xs[pow(2, i)]);
	}

	for(auto i : vector<int>{1,3,5,7})
	{
		ckks->evaluator->multiply_const(x, a[i], xs[i]);
		ckks->evaluator->rescale_to_next_inplace(xs[i]);
	}

	for(auto i : vector<int>{3,7})
	{
		ckks->evaluator->multiply_inplace(xs[i], xs[2]);
		ckks->evaluator->relinearize_inplace(xs[i], relin_keys);
		ckks->evaluator->rescale_to_next_inplace(xs[i]);
	}
	for(auto i : vector<int>{6})
	{
		ckks->evaluator->multiply_const(xs[2], a[i], xs[i]);
		ckks->evaluator->rescale_to_next_inplace(xs[i]);
	}
	for(auto i : vector<int>{5,6,7})
	{
		ckks->evaluator->mod_switch_to_inplace(xs[i], xs[4].parms_id());
		ckks->evaluator->multiply_inplace(xs[i], xs[4]);
		ckks->evaluator->relinearize_inplace(xs[i], relin_keys);
		ckks->evaluator->rescale_to_next_inplace(xs[i]);
	}
	
	for(auto i : vector<int>{2,4})
	{
		ckks->evaluator->multiply_const_inplace(xs[i], a[i]);
		ckks->evaluator->rescale_to_next_inplace(xs[i]);
	}
	auto parm_id = xs[degree-1].parms_id();
	auto scale = xs[degree-1].scale();
	ckks->encoder->encode(a[0], parm_id, scale, pt);
	ckks->encryptor->encrypt(pt, xs[0]);
	for(int i = 0; i < degree; i++)
	{
		ckks->evaluator->mod_switch_to_inplace(xs[i], parm_id);
		// ckks->print_chain_index(xs[i], "xs["+to_string(i));
		xs[i].scale() = scale;
	}
	Ciphertext res;
	ckks->evaluator->add_many(xs, res);
	// ckks->print_dec(res, "res", 1);
	while(r--)
	{
		ckks->evaluator->square_inplace(res);
		ckks->evaluator->relinearize_inplace(res, *(ckks->relin_keys));
		ckks->evaluator->rescale_to_next_inplace(res);
	}
	// ckks->print_chain_index(res, "res");
	// ckks->print_dec(res, "res", 10);
	// func_timer.elapsed("cipher_exp");

	return res;
}

Ciphertext cipher_exp_5(CKKSEvaluator* ckks, Ciphertext& x)  // x in [-pow(2,r)*target, 0] 消耗r+3层
{
	Rectime func_timer;
	func_timer.start();
	// ckks->print_chain_index(x, "x");
	// ckks->print_dec(x, "x", 10);
	int r = 1;
	// ckks->evaluator->multiply_const_inplace(x, 1.0 / pow(2,r));
	// ckks->evaluator->rescale_to_next_inplace(x);
	x.scale() *= pow(2,r);
	int degree = 8;
	// ckks->print_chain_index(x, "x");
	// ckks->print_dec(x, "x", 1);
	vector<double> a{0.9999176045120705, 0.9987416303869157, 0.4953285812872865, 0.159281595830637, 0.03551592614263832, 0.00533666218566905, 0.00048218432821377466, 1.9546405829954175e-05};
					//x/(2^r) in [-5,0] approx: 0.9999176045120705, 0.9987416303869157, 0.4953285812872865, 0.159281595830637, 0.03551592614263832, 0.00533666218566905, 0.00048218432821377466, 1.9546405829954175e-05
					//x/(2^r) in [-4,0] approx: 0.9999791667046215, 0.9996057712501535, 0.49819170081449066, 0.16314976088485714, 0.038088652098859546, 0.006229550870180332, 0.0006372698915844369, 3.018992965556069e-05
					//x/(2^r) in [-3,0] approx: 0.9999968153132669, 0.9999203314486449, 0.4995181577601474, 0.1654360877641768, 0.04003357168254322, 0.007095152387352851, 0.000830588124366306, 4.729606626830923e-05
					//x/(2^r) in [-2,0] approx: 0.999999807711439, 0.9999928438309771, 0.4999357692100727, 0.166424156797964, 0.041193827292401716, 0.00781230587260862, 0.0010544638554755939, 7.517160876064213e-05
	/*
	x0 = a0
	x1 = a1 * x
	x2 = a2 * x^2
	x3 = a3*x * x^2
	x4 = a4 * x^4
	x5 = a5*x * x^4
	x6 = a6*x^2 * x^4
	x7 = a7*x * x^2 * x^4
	*/
	auto relin_keys = *(ckks->relin_keys);
	vector<Ciphertext> xs(degree);
	xs[1] = x;
	Plaintext pt;
	for(int i = 1; i <= 2; i++)
	{
		ckks->evaluator->square(xs[pow(2, i-1)], xs[pow(2, i)]);
		ckks->evaluator->relinearize_inplace(xs[pow(2, i)], relin_keys);
		ckks->evaluator->rescale_to_next_inplace(xs[pow(2, i)]);
	}

	for(auto i : vector<int>{1,3,5,7})
	{
		ckks->evaluator->multiply_const(x, a[i], xs[i]);
		ckks->evaluator->rescale_to_next_inplace(xs[i]);
	}

	for(auto i : vector<int>{3,7})
	{
		ckks->evaluator->multiply_inplace(xs[i], xs[2]);
		ckks->evaluator->relinearize_inplace(xs[i], relin_keys);
		ckks->evaluator->rescale_to_next_inplace(xs[i]);
	}
	for(auto i : vector<int>{6})
	{
		ckks->evaluator->multiply_const(xs[2], a[i], xs[i]);
		ckks->evaluator->rescale_to_next_inplace(xs[i]);
	}
	for(auto i : vector<int>{5,6,7})
	{
		ckks->evaluator->mod_switch_to_inplace(xs[i], xs[4].parms_id());
		ckks->evaluator->multiply_inplace(xs[i], xs[4]);
		ckks->evaluator->relinearize_inplace(xs[i], relin_keys);
		ckks->evaluator->rescale_to_next_inplace(xs[i]);
	}
	
	for(auto i : vector<int>{2,4})
	{
		ckks->evaluator->multiply_const_inplace(xs[i], a[i]);
		ckks->evaluator->rescale_to_next_inplace(xs[i]);
	}
	auto parm_id = xs[degree-1].parms_id();
	auto scale = xs[degree-1].scale();
	ckks->encoder->encode(a[0], parm_id, scale, pt);
	ckks->encryptor->encrypt(pt, xs[0]);
	for(int i = 0; i < degree; i++)
	{
		ckks->evaluator->mod_switch_to_inplace(xs[i], parm_id);
		// ckks->print_chain_index(xs[i], "xs["+to_string(i));
		xs[i].scale() = scale;
	}
	Ciphertext res;
	ckks->evaluator->add_many(xs, res);
	// ckks->print_dec(res, "res", 1);
	while(r--)
	{
		ckks->evaluator->square_inplace(res);
		ckks->evaluator->relinearize_inplace(res, *(ckks->relin_keys));
		ckks->evaluator->rescale_to_next_inplace(res);
	}
	// ckks->print_chain_index(res, "res");
	// ckks->print_dec(res, "res", 10);
	// func_timer.elapsed("cipher_exp");
	return res;
}

Ciphertext cipher_inverse(CKKSEvaluator* ckks, Ciphertext x, double scl, int iter) // 0 < x < scl 消耗iter+1层
{
	Rectime func_timer;
	func_timer.start();
	Plaintext pt;
	tensor ten;
	ckks->decryptor->decrypt(x, pt);
	ckks->encoder->decode(pt, ten);
	// cout << *max_element(ten.begin(), ten.end()) << endl;
	// cout << *min_element(ten.begin(), ten.end()) << endl;
	// auto pram_id = x.parms_id();
	// auto scale = x.scale();
	// ckks->evaluator->multiply_const_inplace(x, 0.01);
	// ckks->evaluator->rescale_to_next_inplace(x);
	// cout << scl << endl;
	// ckks->print_chain_index(x, "x");
	// ckks->print_dec(x, "x", 1);
	x.scale() *= scl;
	// ckks->print_chain_index(x, "x");
	// ckks->print_dec(x, "x", 1);
	/*
	# 0 < x < 1 NEXUS's approximation x 越接近 1 越准确
def inverse(x, iter):
	y = x - 1
	y = -y
	tmp = y + 1
	res = tmp
	# res = 2-x
	for i in range(iter):
		# res = res * ((1-x)^(2^r)+1)
		y = y ** 2
		tmp = y + 1
		res *= tmp
	return res
	*/
	Ciphertext y, tmp, res;
	Plaintext one;
	ckks->encoder->encode(1.0, x.parms_id(), x.scale(), one);
	ckks->evaluator->sub_plain(x, one, y);
	ckks->evaluator->negate_inplace(y);
	ckks->evaluator->add_plain(y, one, tmp);
	res = tmp;
	for (int i = 0; i < iter; i++) {
		ckks->evaluator->square_inplace(y);
		ckks->evaluator->relinearize_inplace(y, *(ckks->relin_keys));
		ckks->evaluator->rescale_to_next_inplace(y);
		// ckks->print_chain_index(y, "y");
		ckks->encoder->encode(1.0, y.parms_id(), y.scale(), one);
		ckks->evaluator->add_plain(y, one, tmp);
		ckks->evaluator->mod_switch_to_inplace(res, tmp.parms_id());
		ckks->evaluator->multiply_inplace(res, tmp);
		ckks->evaluator->relinearize_inplace(res, *(ckks->relin_keys));
		ckks->evaluator->rescale_to_next_inplace(res);
	}
	// ckks->print_dec(res, "res", 10);
	// ckks->evaluator->multiply_const_inplace(res, 0.01);
	// ckks->evaluator->rescale_to_next_inplace(res);
	// ckks->print_chain_index(res, "res");
	res.scale() *= scl;
	// ckks->print_chain_index(res, "res");
	// func_timer.elapsed("cipher_inverse");
	return res;
}

Ciphertext cipher_sqrt(CKKSEvaluator* ckks, Ciphertext x) // sqrt(1+x) = 1+x/2-x^2/8 |x| < 1 消耗2层 
{
	Ciphertext x2;
	Ciphertext res = x;
	ckks->evaluator->square(x, x2);
	ckks->evaluator->relinearize_inplace(x2, *(ckks->relin_keys));
	ckks->evaluator->rescale_to_next_inplace(x2);
	ckks->evaluator->multiply_const_inplace(x2, -1.0/8);
	ckks->evaluator->rescale_to_next_inplace(x2);

	ckks->evaluator->multiply_const_inplace(res, 0.5);
	ckks->evaluator->rescale_to_next_inplace(res);
	ckks->evaluator->multiply_const_inplace(res, 1.0);
	ckks->evaluator->rescale_to_next_inplace(res);

	ckks->evaluator->add_const_inplace(res, 1.0);
	ckks->evaluator->mod_switch_to_inplace(res, x2.parms_id());
	res.scale() = x2.scale();
	ckks->evaluator->add_inplace(res, x2);
	return res;
}

uint64_t get_modulus(CKKSEvaluator* ckks, Ciphertext &x, int k)
{
	const vector<Modulus> &modulus = ckks->context->get_context_data(x.parms_id())->parms().coeff_modulus();
	int sz = modulus.size();
	return modulus[sz - k].value();
}
void eval_odd_deg9_poly(CKKSEvaluator* ckks, vector<double> &a, Ciphertext &x, Ciphertext &dest)
{
/*
      (polyeval/odd9.h)
      P(x) = a9 x^9 + a7 x^7 + a5 x^5 + a3 x^3 + a1 x

      T1 = (a3 + a5 x^2) x^3
      T2 = (a7 x + a9 x^3) x^6
      T3 = a1 x
      P(x) = T1 + T2 + T3

      Depth=4, #Muls=5

      Exactly what babystep_giantstep would do, but written explicitly to optimize

      ###

      -> Errorless Polynomial Evaluation (3.2. of https://eprint.iacr.org/2020/1203)
      GOAL: evaluate a polynomial exactly so no need to stabilize and lose precision
      (x at level L and scale D --> P(x) at level L-4 and scale D)
      it's possible to do this exactly for polyeval as (x,x2,x3,x6) determine the scale D_L for each involved level L:
      (assume the primes at levels L to L-4 are p, q, r, s)

      level       ctx       scale (D_l)
      ==================================
        L          x          D
        L-1        x2         D^2 / p
        L-2        x3         D^3 / pq
        L-3        x6         D^6 / p^2 q^2 r

      Just by encoding constants at different scales we can make every ctx at level l be at scale D_l
      (not possible in general, e.g. rescale(x2*x2) produces L-2 ciphertext with scale D^4/ppq)
      (to fix this we would use the Adjust op. that multiplies ctx by constants and Algo 3 for primes from
     https://eprint.iacr.org/2020/1118)

      Now we know that sc(P(x)) should be D, so we recursively go back to compute the scales for each coefficient
      sc(T1)=sc(T2)=sc(T3)=sc(P(x))=D

      T3:
          sc(a1) = q (should be p but it gets multiplied with modswitched x)

      T2:
          sc(x^6) = D^6 / p^2 q^2 r, so sc(a7*x) = sc(a9*x^3) = p^2 q^2 r s / D^5
          next, sc(a7) = p^2 q^3 r s / D^6
          similarly, sc(a9) = p^3 q^3 r^2 s / D^8

      T1:
          sc(x^3) = D^3 / pq
          implying sc(a3) = pqr / D^2 and also sc(a5*x^2) = pqr / D^2
          as sc(x^2) = D^2 / p this implies sc(a5) = p^2 q^2 r / D^4
  */
  // chrono::high_resolution_clock::time_point time_start, time_end;
  // time_start = high_resolution_clock::now();
	double D = x.scale();  // maybe not init_scale but preserved

	uint64_t p = get_modulus(ckks, x, 1);
	uint64_t q = get_modulus(ckks, x, 2);
	uint64_t r = get_modulus(ckks, x, 3);
	uint64_t s = get_modulus(ckks, x, 4);
	uint64_t t = get_modulus(ckks, x, 5);

	p = q;
	q = r;
	r = s;
	s = t;

	vector<double> a_scales(10);
	a_scales[1] = q;
	a_scales[3] = (double)p / D * q / D * r;
	a_scales[5] = (double)p / D * p / D * q / D * q / D * r;
	a_scales[7] = (double)p / D * p / D * q / D * q / D * q / D * r / D * s;
	a_scales[9] = (double)p / D * p / D * p / D * q / D * q / D * q / D * r / D * r / D * s;

	///////////////////////////////////////////////
	Ciphertext x2, x3, x6;

	ckks->evaluator->square(x, x2);
	ckks->evaluator->relinearize_inplace(x2, *(ckks->relin_keys));
	ckks->evaluator->rescale_to_next_inplace(x2);  // L-1

	ckks->evaluator->mod_switch_to_next_inplace(x);  // L-1
	ckks->evaluator->multiply(x2, x, x3);
	ckks->evaluator->relinearize_inplace(x3, *(ckks->relin_keys));
	ckks->evaluator->rescale_to_next_inplace(x3);  // L-2

	ckks->evaluator->square(x3, x6);
	ckks->evaluator->relinearize_inplace(x6, *(ckks->relin_keys));
	ckks->evaluator->rescale_to_next_inplace(x6);  // L-3

	Plaintext a1, a3, a5, a7, a9;

	// Build T1
	Ciphertext T1;
	double a5_scale = D / x2.scale() * p / x3.scale() * q;
	ckks->encoder->encode(a[5], x2.parms_id(), a5_scale, a5);  // L-1
	ckks->evaluator->multiply_plain(x2, a5, T1);
	ckks->evaluator->rescale_to_next_inplace(T1);  // L-2

	// Update: using a_scales[3] is only approx. correct, so we directly use T1.scale()
	ckks->encoder->encode(a[3], T1.parms_id(), T1.scale(), a3);  // L-2

	ckks->evaluator->add_plain_inplace(T1, a3);  // L-2

	ckks->evaluator->multiply_inplace(T1, x3);
	ckks->evaluator->relinearize_inplace(T1, *(ckks->relin_keys));
	ckks->evaluator->rescale_to_next_inplace(T1);  // L-3

	// Build T2
	Ciphertext T2;
	Plaintext a9_switched;
	double a9_scale = D / x3.scale() * r / x6.scale() * q;
	ckks->encoder->encode(a[9], x3.parms_id(), a9_scale, a9);  // L-2
	ckks->evaluator->multiply_plain(x3, a9, T2);
	ckks->evaluator->rescale_to_next_inplace(T2);  // L-3

	Ciphertext a7x;
	double a7_scale = T2.scale() / x.scale() * p;
	ckks->encoder->encode(a[7], x.parms_id(), a7_scale, a7);  // L-1 (x was modswitched)
	ckks->evaluator->multiply_plain(x, a7, a7x);
	ckks->evaluator->rescale_to_next_inplace(a7x);               // L-2
	ckks->evaluator->mod_switch_to_inplace(a7x, T2.parms_id());  // L-3

	double mid_scale = (T2.scale() + a7x.scale()) / 2;
	T2.scale() = a7x.scale() = mid_scale;  // this is the correct scale now, need to set it still to avoid SEAL assert
	ckks->evaluator->add_inplace(T2, a7x);       // L-3
	ckks->evaluator->multiply_inplace(T2, x6);
	ckks->evaluator->relinearize_inplace(T2, *(ckks->relin_keys));
	ckks->evaluator->rescale_to_next_inplace(T2);  // L-4

	// Build T3
	Ciphertext T3;
	ckks->encoder->encode(a[1], x.parms_id(), p, a1);  // L-1 (x was modswitched)
	ckks->evaluator->multiply_plain(x, a1, T3);
	ckks->evaluator->rescale_to_next_inplace(T3);  // L-2

	// T1, T2 and T3 should be on the same scale up to floating point
	// but we still need to set them manually to avoid SEAL assert
	double mid3_scale = (T1.scale() + T2.scale() + T3.scale()) / 3;
	T1.scale() = T2.scale() = T3.scale() = mid3_scale;

	dest = T2;
	ckks->evaluator->mod_switch_to_inplace(T1, dest.parms_id());  // L-4
	ckks->evaluator->add_inplace(dest, T1);
	ckks->evaluator->mod_switch_to_inplace(T3, dest.parms_id());  // L-4
	ckks->evaluator->add_inplace(dest, T3);

	/////////////////////////////////////////
	// it should be ==D but we don't stabilize if it's not, D' != D is ok
	// the goal was to make T1+T2+T3 work with minimal loss in precision
	// time_end = high_resolution_clock::now();
	// cout << "Poly eval took " << duration_cast<milliseconds>(time_end - time_start).count() << " ms" << endl;
}


Ciphertext cipher_sign(CKKSEvaluator* ckks, Ciphertext x, int d_g, int d_f, double sgn_factor)
{
	vector<double> F4_COEFFS = {0, 315, 0, -420, 0, 378, 0, -180, 0, 35};
	// should be divided by (1 << 7)
	int f4_scale = (1 << 7);
	vector<double> G4_COEFFS = {0, 5850, 0, -34974, 0, 97015, 0, -113492, 0, 46623};
	// should be divided by (1 << 10)
	int g4_scale = (1 << 10);
	// Compute sign function coefficients
	vector<double> f4_coeffs = F4_COEFFS;
	vector<double> g4_coeffs = G4_COEFFS;
	vector<double> f4_coeffs_last(10, 0.0);
	vector<double> g4_coeffs_last(10, 0.0);
  
	for (int i = 0; i <= 9; i++) {
	  f4_coeffs[i] /= f4_scale;
	  f4_coeffs_last[i] = f4_coeffs[i] * sgn_factor;
  
	  g4_coeffs[i] /= g4_scale;
	  g4_coeffs_last[i] = g4_coeffs[i] * sgn_factor;
	}
	
	Ciphertext dest = x;	
	for (int i = 0; i < d_g; i++) {
		// ckks->print_chain_index(dest, "dest");
	  if (i == d_g - 1) {
		eval_odd_deg9_poly(ckks, g4_coeffs_last, dest, dest);
	  } else {
		eval_odd_deg9_poly(ckks, g4_coeffs, dest, dest);
	  }
	}
	for (int i = 0; i < d_f; i++) {
		// ckks->print_chain_index(dest, "dest");
	  if (i == d_f - 1) {
		eval_odd_deg9_poly(ckks, f4_coeffs_last, dest, dest);
	  } else {
		eval_odd_deg9_poly(ckks, f4_coeffs, dest, dest);
	  }
	}
	return dest;
}
