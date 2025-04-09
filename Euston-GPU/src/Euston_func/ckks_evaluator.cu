#include "ckks_evaluator.cuh"


using namespace std;
using namespace chrono;

double SCALE = pow(2.0, 40);

void CKKSEvaluator::print_chain_index(PhantomCiphertext ct, string ct_name)
{
	cout << fixed << setprecision(10);
	cout << "    + Modulus chain index for " + ct_name << " : "
         << ct.chain_index() << endl;
	cout << "    + Scale of " + ct_name << " : "  << log2(ct.scale()) << " bits" << endl;
}

void CKKSEvaluator::print_dec(PhantomCiphertext ct, string des, int num)
{
	cout << fixed << setprecision(6);
	PhantomPlaintext pt;
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
double CKKSEvaluator::calculate_errors(PhantomCiphertext ct, tensor list)
{
	PhantomPlaintext pt;
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
    size_t logN = 13;
    size_t poly_modulus_degree = 1 << logN;

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, coeffs));

    auto context = make_shared<PhantomContext>(parms);
	auto encoder = make_shared<PhantomCKKSEncoder>(*context);
	shared_ptr<PhantomSecretKey> secret_key = make_shared<PhantomSecretKey>(*context);

    auto public_key = make_shared<PhantomPublicKey>(secret_key->gen_publickey(*context));
	auto relin_keys = make_shared<PhantomRelinKey>(secret_key->gen_relinkey(*context));
    auto galois_keys = make_shared<PhantomGaloisKey>();
    auto galois_keys_allrotate = make_shared<PhantomGaloisKey>();

	// Instantiate the component classes
	auto ckks_encoder = make_shared<Encoder>(context, encoder);
	auto encryptor = make_shared<Encryptor>(context, public_key);
	auto evaluator = make_shared<Evaluator>(context, encoder);
	auto decryptor = make_shared<Decryptor>(context, secret_key);


    CKKSEvaluator ckks_evaluator(context, encryptor, decryptor, ckks_encoder, evaluator, relin_keys, galois_keys, galois_keys_allrotate, SCALE);
    return ckks_evaluator;
}

CKKSEvaluator construct_ckks_evaluator_nonlinear(long logN, vector<int> coeffs)
{
    size_t poly_modulus_degree = 1 << logN;

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, coeffs));

    auto context = make_shared<PhantomContext>(parms);
	auto encoder = make_shared<PhantomCKKSEncoder>(*context);
	shared_ptr<PhantomSecretKey> secret_key = make_shared<PhantomSecretKey>(*context);

    auto public_key = make_shared<PhantomPublicKey>(secret_key->gen_publickey(*context));
	auto relin_keys = make_shared<PhantomRelinKey>(secret_key->gen_relinkey(*context));
    auto galois_keys = make_shared<PhantomGaloisKey>();
    auto galois_keys_allrotate = make_shared<PhantomGaloisKey>();

	// Instantiate the component classes
	shared_ptr<Encoder> ckks_encoder = make_shared<Encoder>(context, encoder);
	shared_ptr<Encryptor> encryptor = make_shared<Encryptor>(context, public_key);
	shared_ptr<Evaluator> evaluator = make_shared<Evaluator>(context, encoder);
	shared_ptr<Decryptor> decryptor = make_shared<Decryptor>(context, secret_key);


    CKKSEvaluator ckks_evaluator(context, encryptor, decryptor, ckks_encoder, evaluator, relin_keys, galois_keys, galois_keys_allrotate, SCALE);
    return ckks_evaluator;
}

void switch_ckks_evaluator(PhantomCiphertext& ct, CKKSEvaluator* ckks1, CKKSEvaluator* ckks2)
{
	PhantomPlaintext pt;
	tensor ten;
	ckks1->decryptor->decrypt(ct, pt);
	ckks1->encoder->decode(pt, ten);
	ckks2->encoder->encode(ten, ckks2->scale, pt);
	ckks2->encryptor->encrypt(pt, ct);
}

__global__ void kernel_compress_ciphertext(uint64_t *plain_data, size_t plain_scale, size_t degree, size_t coeff_modulus_size, const DModulus *moduli, const double *values) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < degree) {
        auto coeffd = std::round(values[idx] * plain_scale);
        bool is_negative = std::signbit(coeffd);
        auto coeffu = static_cast<std::uint64_t>(std::fabs(coeffd));

        if (is_negative) {
			for (std::size_t j = 0; j < coeff_modulus_size; j++) {
			  	plain_data[idx + (j * degree)] = negate_uint64_mod(
				barrett_reduce_uint64_uint64(coeffu, moduli[j].value(), moduli[j].const_ratio()[1]), moduli[j].value());
			}
		} 
		else {
			for (std::size_t j = 0; j < coeff_modulus_size; j++) {
			  	plain_data[idx + (j * degree)] = barrett_reduce_uint64_uint64(coeffu, moduli[j].value(), moduli[j].const_ratio()[1]);
			}
		}
    }
}

__global__ void kernel_negacyclic_shift(const uint64_t *cipher_data, const size_t cipher_count, const uint64_t coeff_count, const size_t mod_count, const int shift, const DModulus *moduli, uint64_t *dest_data)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < cipher_count * mod_count * coeff_count) {
        if (shift == 0) {
            dest_data[idx] = cipher_data[idx];
            return;
        }

        size_t i = idx / (mod_count * coeff_count);
        size_t j = (idx / coeff_count) % mod_count;
        size_t k = idx % coeff_count;
        size_t mask = coeff_count - 1;
        uint64_t modulus_value = moduli[j].value();

        size_t index = (shift + k) & mask;
        size_t result_index = i * mod_count * coeff_count + j * coeff_count + index;
        if (cipher_data[idx] == 0 || ((shift + k) & coeff_count) == 0) {
            dest_data[result_index] = cipher_data[idx];
        } else {
            dest_data[result_index] = modulus_value - cipher_data[idx];
        }
    }
}

void encrypting_compress(const CKKSEvaluator* ckks, const tensor& values, PhantomCiphertext &ct)
{
	size_t plain_scale = 10000000000;

	auto &context_data = ckks->context->first_context_data();
	auto param = context_data.parms();
	auto moduli = ckks->context->gpu_rns_tables().modulus();
	auto coeff_modulus_size = param.coeff_modulus().size();
	auto poly_modulus_degree = param.poly_modulus_degree();

	const phantom::util::cuda_stream_wrapper &stream_wrapper = *phantom::util::global_variables::default_stream;
	const auto &stream = stream_wrapper.get_stream();

	PhantomPlaintext p;
	p.resize(coeff_modulus_size, poly_modulus_degree, stream);

	auto gpu_values = make_cuda_auto_ptr<double>(values.size(), stream);
	cudaMemcpyAsync(gpu_values.get(), values.data(), values.size() * sizeof(double), cudaMemcpyHostToDevice, stream);

	kernel_compress_ciphertext<<<poly_modulus_degree / blockDimGlb.x, blockDimGlb, 0, stream>>>(
		p.data(), plain_scale, poly_modulus_degree, coeff_modulus_size, moduli, gpu_values.get());

	// Transform polynomials to the NTT domain
	nwt_2d_radix8_forward_inplace(p.data(), ckks->context->gpu_rns_tables(), coeff_modulus_size, 0, stream);

	// Update plaintext parameters
	p.parms_id() = context_data.parms().parms_id();
	p.set_chain_index(context_data.chain_index());
	p.scale() = plain_scale;

	// Create a ciphertext encrypting zero
	PhantomPlaintext zero_pt;
	PhantomCiphertext zero;
	ckks->encoder->encode(0.0, plain_scale, zero_pt);
	ckks->encryptor->encrypt(zero_pt, zero);

	// Encrypt the plaintext
	ckks->evaluator->add_plain(zero, p, ct);
}

vector<PhantomCiphertext> encrypting_decompress(const CKKSEvaluator* ckks, const PhantomCiphertext &encrypted) 
{
	auto N = ckks->degree;
	uint32_t logN = ceil(log2(N));

	vector<PhantomCiphertext> temp;
	temp.push_back(encrypted);

	PhantomCiphertext tempctxt_rotated;
	PhantomCiphertext tempctxt_shifted;
	PhantomCiphertext tempctxt_rotatedshifted;

	for (uint32_t i = 0; i < logN; i++) {
		vector<PhantomCiphertext> newtemp(temp.size() << 1);

		uint32_t galois_elt = ckks->galois_elts[i];
		int index_raw = (N << 1) - (1 << i);
		int index = (index_raw * galois_elt) % (N << 1);

		for (uint32_t a = 0; a < temp.size(); a++) {
			ckks->evaluator->apply_galois(temp[a], galois_elt, *(ckks->galois_keys), tempctxt_rotated);  // sub
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

void multiply_power_of_x(const CKKSEvaluator* ckks, PhantomCiphertext &encrypted, PhantomCiphertext &destination, int index) 
{
	auto context = ckks->context;
	auto coeff_count = ckks->degree;
	auto param = context->get_context_data(encrypted.params_id()).parms();
	auto moduli = param.coeff_modulus();
	auto coeff_mod_count = param.coeff_modulus().size();
	auto encrypted_count = encrypted.size();
	auto rns_coeff_count = coeff_count * coeff_mod_count;

	const auto &stream = phantom::util::global_variables::default_stream->get_stream();

	destination = encrypted;
	ckks->evaluator->transform_from_ntt_inplace(destination);

	auto dest_data = new uint64_t[rns_coeff_count * encrypted_count];
	auto dest_data_copy = new uint64_t[rns_coeff_count * encrypted_count];
	cudaMemcpyAsync(dest_data, destination.data(), encrypted_count * rns_coeff_count * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
	std::copy(dest_data, dest_data + rns_coeff_count * encrypted_count, dest_data_copy);

	for (int i = 0; i < encrypted_count; i++)
	{
		for (int j = 0; j < coeff_mod_count; j++)
		{
			uint64_t *poly = dest_data_copy + i * rns_coeff_count + j * coeff_count;
			uint64_t *result = dest_data + i * rns_coeff_count + j * coeff_count;

			uint64_t index_raw = index;
			uint64_t coeff_count_mod_mask = static_cast<uint64_t>(coeff_count) - 1;
			for (size_t k = 0; k < coeff_count; k++, poly++, index_raw++)
			{
				uint64_t index = index_raw & coeff_count_mod_mask;
				if (!(index_raw & static_cast<uint64_t>(coeff_count)) || !*poly)
				{
					result[index] = *poly;
				}
				else
				{
					result[index] = moduli[j].value() - *poly;
				}
			}
		}
	}

	cudaMemcpyAsync(destination.data(), dest_data, encrypted_count * rns_coeff_count * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);

	delete dest_data;
	delete dest_data_copy;

	ckks->evaluator->transform_to_ntt_inplace(destination);
}


// size_t cal_size(PhantomCiphertext ct, string description)
// {
// 	vector<seal::seal_byte> rece_bytes(ct.save_size());
// 	size_t siz = ct.save(rece_bytes.data(), rece_bytes.size());
// 	cout << description;
// 	cout << siz / 1024.0 / 1024.0 << " MB" << endl; 
// 	return siz;
// }


PhantomCiphertext cipher_exp(CKKSEvaluator* ckks, PhantomCiphertext& x)  // x in [-pow(2,r)*target, 0] 消耗r+3层
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
	auto relin_keys = ckks->relin_keys;
	vector<PhantomCiphertext> xs(degree);
	xs[1] = x;
	PhantomPlaintext pt;
	for(int i = 1; i <= 2; i++)
	{
		ckks->evaluator->square(xs[pow(2, i-1)], xs[pow(2, i)]);
		ckks->evaluator->relinearize_inplace(xs[pow(2, i)], *relin_keys);
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
		ckks->evaluator->relinearize_inplace(xs[i], *relin_keys);
		ckks->evaluator->rescale_to_next_inplace(xs[i]);
	}
	for(auto i : vector<int>{6})
	{
		ckks->evaluator->multiply_const(xs[2], a[i], xs[i]);
		ckks->evaluator->rescale_to_next_inplace(xs[i]);
	}
	for(auto i : vector<int>{5,6,7})
	{
		ckks->evaluator->mod_switch_to_inplace(xs[i], xs[4].params_id());
		ckks->evaluator->multiply_inplace(xs[i], xs[4]);
		ckks->evaluator->relinearize_inplace(xs[i], *relin_keys);
		ckks->evaluator->rescale_to_next_inplace(xs[i]);
	}
	
	for(auto i : vector<int>{2,4})
	{
		ckks->evaluator->multiply_const_inplace(xs[i], a[i]);
		ckks->evaluator->rescale_to_next_inplace(xs[i]);
	}
	auto parm_id = xs[degree-1].params_id();
	auto scale = xs[degree-1].scale();
	ckks->encoder->encode(a[0], parm_id, scale, pt);
	ckks->encryptor->encrypt(pt, xs[0]);
	for(int i = 0; i < degree; i++)
	{
		ckks->evaluator->mod_switch_to_inplace(xs[i], parm_id);
		// ckks->print_chain_index(xs[i], "xs["+to_string(i));
		xs[i].scale() = scale;
	}
	PhantomCiphertext res;
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

PhantomCiphertext cipher_exp_5(CKKSEvaluator* ckks, PhantomCiphertext& x)  // x in [-pow(2,r)*target, 0] 消耗r+3层
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
	auto relin_keys = ckks->relin_keys;
	vector<PhantomCiphertext> xs(degree);
	xs[1] = x;
	PhantomPlaintext pt;
	for(int i = 1; i <= 2; i++)
	{
		ckks->evaluator->square(xs[pow(2, i-1)], xs[pow(2, i)]);
		ckks->evaluator->relinearize_inplace(xs[pow(2, i)], *relin_keys);
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
		ckks->evaluator->relinearize_inplace(xs[i], *relin_keys);
		ckks->evaluator->rescale_to_next_inplace(xs[i]);
	}
	for(auto i : vector<int>{6})
	{
		ckks->evaluator->multiply_const(xs[2], a[i], xs[i]);
		ckks->evaluator->rescale_to_next_inplace(xs[i]);
	}
	for(auto i : vector<int>{5,6,7})
	{
		ckks->evaluator->mod_switch_to_inplace(xs[i], xs[4].params_id());
		ckks->evaluator->multiply_inplace(xs[i], xs[4]);
		ckks->evaluator->relinearize_inplace(xs[i], *relin_keys);
		ckks->evaluator->rescale_to_next_inplace(xs[i]);
	}
	
	for(auto i : vector<int>{2,4})
	{
		ckks->evaluator->multiply_const_inplace(xs[i], a[i]);
		ckks->evaluator->rescale_to_next_inplace(xs[i]);
	}
	auto parm_id = xs[7].params_id();
	auto scale = xs[7].scale();
	ckks->encoder->encode(a[0], scale, pt);
	ckks->encryptor->encrypt(pt, xs[0]);
	for(int i = 0; i < degree-1; i++)
	{
		ckks->evaluator->mod_switch_to_inplace(xs[i], parm_id);
		xs[i].scale() = scale;
	}
	PhantomCiphertext res;
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

PhantomCiphertext cipher_inverse(CKKSEvaluator* ckks, PhantomCiphertext x, double scl, int iter) // 0 < x < scl 消耗iter+1层
{
	Rectime func_timer;
	func_timer.start();
	PhantomPlaintext pt;
	tensor ten;
	ckks->decryptor->decrypt(x, pt);
	ckks->encoder->decode(pt, ten);
	// cout << *max_element(ten.begin(), ten.end()) << endl;
	// cout << *min_element(ten.begin(), ten.end()) << endl;
	// auto pram_id = x.params_id();
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
	PhantomCiphertext y, tmp, res;
	PhantomPlaintext one;
	ckks->encoder->encode(1.0, x.params_id(), x.scale(), one);
	ckks->evaluator->sub_plain(x, one, y);
	ckks->evaluator->negate_inplace(y);
	ckks->evaluator->add_plain(y, one, tmp);
	res = tmp;
	for (int i = 0; i < iter; i++) {
		ckks->evaluator->square_inplace(y);
		ckks->evaluator->relinearize_inplace(y, *(ckks->relin_keys));
		ckks->evaluator->rescale_to_next_inplace(y);
		// ckks->print_chain_index(y, "y");
		ckks->encoder->encode(1.0, y.params_id(), y.scale(), one);
		ckks->evaluator->add_plain(y, one, tmp);
		ckks->evaluator->mod_switch_to_inplace(res, tmp.params_id());
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

PhantomCiphertext cipher_sqrt(CKKSEvaluator* ckks, PhantomCiphertext x) // sqrt(1+x) = 1+x/2-x^2/8 |x| < 1 消耗2层 
{
	PhantomCiphertext x2;
	PhantomCiphertext res = x;
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
	ckks->evaluator->mod_switch_to_inplace(res, x2.params_id());
	res.scale() = x2.scale();
	ckks->evaluator->add_inplace(res, x2);
	return res;
}

uint64_t get_modulus(CKKSEvaluator* ckks, PhantomCiphertext &x, int k) {
	const vector<phantom::arith::Modulus> &modulus = ckks->context->get_context_data(x.params_id()).parms().coeff_modulus();
	int sz = modulus.size();
	return modulus[sz - k].value();
  }
void eval_odd_deg9_poly(CKKSEvaluator* ckks, vector<double> &a, PhantomCiphertext &x, PhantomCiphertext &dest)
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
      (not possible in general, e.g. rescale(x2*x2) produces L-2 PhantomCiphertext with scale D^4/ppq)
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
	PhantomCiphertext x2, x3, x6;

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

	PhantomPlaintext a1, a3, a5, a7, a9;

	// Build T1
	PhantomCiphertext T1;
	double a5_scale = D / x2.scale() * p / x3.scale() * q;
	ckks->encoder->encode(a[5], x2.params_id(), a5_scale, a5);  // L-1
	ckks->evaluator->multiply_plain(x2, a5, T1);
	ckks->evaluator->rescale_to_next_inplace(T1);  // L-2

	// Update: using a_scales[3] is only approx. correct, so we directly use T1.scale()
	ckks->encoder->encode(a[3], T1.params_id(), T1.scale(), a3);  // L-2

	ckks->evaluator->add_plain_inplace(T1, a3);  // L-2

	ckks->evaluator->multiply_inplace(T1, x3);
	ckks->evaluator->relinearize_inplace(T1, *(ckks->relin_keys));
	ckks->evaluator->rescale_to_next_inplace(T1);  // L-3

	// Build T2
	PhantomCiphertext T2;
	PhantomPlaintext a9_switched;
	double a9_scale = D / x3.scale() * r / x6.scale() * q;
	ckks->encoder->encode(a[9], x3.params_id(), a9_scale, a9);  // L-2
	ckks->evaluator->multiply_plain(x3, a9, T2);
	ckks->evaluator->rescale_to_next_inplace(T2);  // L-3

	PhantomCiphertext a7x;
	double a7_scale = T2.scale() / x.scale() * p;
	ckks->encoder->encode(a[7], x.params_id(), a7_scale, a7);  // L-1 (x was modswitched)
	ckks->evaluator->multiply_plain(x, a7, a7x);
	ckks->evaluator->rescale_to_next_inplace(a7x);               // L-2
	ckks->evaluator->mod_switch_to_inplace(a7x, T2.params_id());  // L-3

	double mid_scale = (T2.scale() + a7x.scale()) / 2;
	T2.scale() = a7x.scale() = mid_scale;  // this is the correct scale now, need to set it still to avoid SEAL assert
	ckks->evaluator->add_inplace(T2, a7x);       // L-3
	ckks->evaluator->multiply_inplace(T2, x6);
	ckks->evaluator->relinearize_inplace(T2, *(ckks->relin_keys));
	ckks->evaluator->rescale_to_next_inplace(T2);  // L-4

	// Build T3
	PhantomCiphertext T3;
	ckks->encoder->encode(a[1], x.params_id(), p, a1);  // L-1 (x was modswitched)
	ckks->evaluator->multiply_plain(x, a1, T3);
	ckks->evaluator->rescale_to_next_inplace(T3);  // L-2

	// T1, T2 and T3 should be on the same scale up to floating point
	// but we still need to set them manually to avoid SEAL assert
	double mid3_scale = (T1.scale() + T2.scale() + T3.scale()) / 3;
	T1.scale() = T2.scale() = T3.scale() = mid3_scale;

	dest = T2;
	ckks->evaluator->mod_switch_to_inplace(T1, dest.params_id());  // L-4
	ckks->evaluator->add_inplace(dest, T1);
	ckks->evaluator->mod_switch_to_inplace(T3, dest.params_id());  // L-4
	ckks->evaluator->add_inplace(dest, T3);

	/////////////////////////////////////////
	// it should be ==D but we don't stabilize if it's not, D' != D is ok
	// the goal was to make T1+T2+T3 work with minimal loss in precision
	// time_end = high_resolution_clock::now();
	// cout << "Poly eval took " << duration_cast<milliseconds>(time_end - time_start).count() << " ms" << endl;
}


PhantomCiphertext cipher_sign(CKKSEvaluator* ckks, PhantomCiphertext x, int d_g, int d_f, double sgn_factor)
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
	
	PhantomCiphertext dest = x;	
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



void Evaluator::sub_inplace_reduced_error(PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2) {
	size_t encrypted1_coeff_modulus_size = encrypted1.coeff_modulus_size();
	size_t encrypted2_coeff_modulus_size = encrypted2.coeff_modulus_size();
  
	if (encrypted1_coeff_modulus_size == encrypted2_coeff_modulus_size) {
	  encrypted1.scale() = encrypted2.scale();
	  sub_inplace(encrypted1, encrypted2);
	  return;
	}
  
	else if (encrypted1_coeff_modulus_size < encrypted2_coeff_modulus_size) {
	  auto &context_data = context->get_context_data(encrypted2.params_id());
	  auto &parms = context_data.parms();
	  auto modulus = parms.coeff_modulus();
	  PhantomCiphertext encrypted2_adjusted;
  
	  double scale_adjust = encrypted1.scale() * static_cast<double>(modulus[encrypted2_coeff_modulus_size - 1].value()) / (encrypted2.scale() * encrypted2.scale());
	  multiply_const(encrypted2, scale_adjust, encrypted2_adjusted);
	  encrypted2_adjusted.scale() = encrypted1.scale() * static_cast<double>(modulus[encrypted2_coeff_modulus_size - 1].value());
	  rescale_to_next_inplace(encrypted2_adjusted);
	  mod_switch_to_inplace(encrypted2_adjusted, encrypted1.params_id());
	  encrypted1.scale() = encrypted2_adjusted.scale();
	  sub_inplace(encrypted1, encrypted2_adjusted);
	}
  
	else {
	  auto &context_data = context->get_context_data(encrypted1.params_id());
	  auto &parms = context_data.parms();
	  auto modulus = parms.coeff_modulus();
	  PhantomCiphertext encrypted1_adjusted;
  
	  double scale_adjust = encrypted2.scale() * static_cast<double>(modulus[encrypted1_coeff_modulus_size - 1].value()) / (encrypted1.scale() * encrypted1.scale());
	  multiply_const(encrypted1, scale_adjust, encrypted1_adjusted);
	  encrypted1_adjusted.scale() = encrypted2.scale() * static_cast<double>(modulus[encrypted1_coeff_modulus_size - 1].value());
	  rescale_to_next_inplace(encrypted1_adjusted);
	  mod_switch_to_inplace(encrypted1_adjusted, encrypted2.params_id());
	  encrypted1_adjusted.scale() = encrypted2.scale();
	  sub(encrypted1_adjusted, encrypted2, encrypted1);
	}
  }
  
  void Evaluator::multiply_inplace_reduced_error(PhantomCiphertext &encrypted1, const PhantomCiphertext &encrypted2, const PhantomRelinKey &relin_keys) {
	size_t encrypted1_coeff_modulus_size = encrypted1.coeff_modulus_size();
	size_t encrypted2_coeff_modulus_size = encrypted2.coeff_modulus_size();
  
	if (encrypted1_coeff_modulus_size == encrypted2_coeff_modulus_size) {
	  encrypted1.scale() = encrypted2.scale();
	  multiply_inplace(encrypted1, encrypted2);
	  relinearize_inplace(encrypted1, relin_keys);
	  return;
	}
  
	else if (encrypted1_coeff_modulus_size < encrypted2_coeff_modulus_size) {
	  auto &context_data = context->get_context_data(encrypted2.params_id());
	  auto &parms = context_data.parms();
	  auto modulus = parms.coeff_modulus();
	  PhantomCiphertext encrypted2_adjusted;
  
	  double scale_adjust = encrypted1.scale() * static_cast<double>(modulus[encrypted2_coeff_modulus_size - 1].value()) / (encrypted2.scale() * encrypted2.scale());
	  multiply_const(encrypted2, scale_adjust, encrypted2_adjusted);
	  encrypted2_adjusted.scale() = encrypted1.scale() * static_cast<double>(modulus[encrypted2_coeff_modulus_size - 1].value());
	  rescale_to_next_inplace(encrypted2_adjusted);
	  mod_switch_to_inplace(encrypted2_adjusted, encrypted1.params_id());
	  encrypted1.scale() = encrypted2_adjusted.scale();
	  multiply_inplace(encrypted1, encrypted2_adjusted);
	}
  
	else {
	  auto &context_data = context->get_context_data(encrypted1.params_id());
	  auto &parms = context_data.parms();
	  auto modulus = parms.coeff_modulus();
	  PhantomCiphertext encrypted1_adjusted;
  
	  double scale_adjust = encrypted2.scale() * static_cast<double>(modulus[encrypted1_coeff_modulus_size - 1].value()) / (encrypted1.scale() * encrypted1.scale());
	  multiply_const(encrypted1, scale_adjust, encrypted1_adjusted);
	  encrypted1_adjusted.scale() = encrypted2.scale() * static_cast<double>(modulus[encrypted1_coeff_modulus_size - 1].value());
	  rescale_to_next_inplace(encrypted1_adjusted);
	  mod_switch_to_inplace(encrypted1_adjusted, encrypted2.params_id());
	  encrypted1_adjusted.scale() = encrypted2.scale();
	  multiply(encrypted1_adjusted, encrypted2, encrypted1);
	}
  
	relinearize_inplace(encrypted1, relin_keys);
  }
  