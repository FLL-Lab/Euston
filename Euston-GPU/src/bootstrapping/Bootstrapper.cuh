#pragma once

#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>

#include "../utils.cuh"
#include "ModularReducer.cuh"

using namespace std;
using namespace nexus;
using namespace phantom;
using namespace phantom::util;

class Bootstrapper {
 public:
  long loge;
  long logn;
  long n;
  long logNh;
  long Nh;
  long L;

  double initial_scale;
  double final_scale;

  long boundary_K;
  long sin_cos_deg;
  long scale_factor;
  long inverse_deg;

  CKKSEvaluator *ckks = nullptr;

  vector<long> slot_vec;
  long slot_index = 0;
  vector<vector<vector<vector<complex<double>>>>> orig_coeffvec, orig_invcoeffvec;
  vector<vector<vector<complex<double>>>> fftcoeff1, fftcoeff2, fftcoeff3;
  vector<vector<vector<complex<double>>>> invfftcoeff1, invfftcoeff2, invfftcoeff3;

  vector<PhantomPlaintext> fftcoeff_plain1, fftcoeff_plain2, invfftcoeff_plain1, invfftcoeff_plain2;

  ModularReducer *mod_reducer;

  Bootstrapper(
      long _loge,
      long _logn,
      long _logNh,
      long _L,
      double _final_scale,
      long _boundary_K,
      long _sin_cos_deg,
      long _scale_factor,
      long _inverse_deg,
      CKKSEvaluator *ckks);

  inline void set_final_scale(double _final_scale) {
    final_scale = _final_scale;
  }

  // Add rotation keys needed in bootstrapping (private function)
  void addLeftRotKeys_Linear_to_vector(vector<int> &gal_steps_vector);
  void addLeftRotKeys_Linear_to_vector_3(vector<int> &gal_steps_vector);

  void addLeftRotKeys_Linear_to_vector_3_other_slots(vector<int> &gal_steps_vector, long other_logn);

  void addLeftRotKeys_Linear_to_vector_one_depth(vector<int> &gal_steps_vector);
  void addLeftRotKeys_Linear_to_vector_one_depth_more_depth(vector<int> &gal_steps_vector);

  // Add rotation keys needed in bootstrapping (public function)
  void addBootKeys(PhantomGaloisKey &gal_keys);
  void addBootKeys_3(PhantomGaloisKey &gal_keys);
  void addBootKeys_other_keys(PhantomGaloisKey &gal_keys, vector<int> &other_keys);
  void addBootKeys_3_other_keys(PhantomGaloisKey &gal_keys, vector<int> &other_keys);

  void addBootKeys_3_other_slots(PhantomGaloisKey &gal_keys, vector<long> &other_logn_vec);
  void addBootKeys_3_other_slots_keys(PhantomGaloisKey &gal_keys, vector<long> &other_logn_vec, vector<int> &other_keys);

  void addBootKeys_hoisting(PhantomGaloisKey &gal_keys);
  void addBootKeys_one_depth(PhantomGaloisKey &gal_keys);
  void addBootKeys_one_depth_more_depth(PhantomGaloisKey &gal_keys);

  void change_logn(long new_logn);

  // Prepare the FFT coefficients
  void genorigcoeff();
  void genfftcoeff_one_depth();
  void genfftcoeff_full_one_depth();
  void geninvfftcoeff_one_depth();
  void generate_LT_coefficient_one_depth();

  void genfftcoeff();
  void genfftcoeff_full();
  void geninvfftcoeff();
  void geninvfftcoeff_full();

  void genfftcoeff_3();
  void geninvfftcoeff_3();
  void generate_LT_coefficient();
  void generate_LT_coefficient_3();

  // Prepare the approximate polynomial
  void prepare_mod_polynomial();

  void subsum(double scale, PhantomCiphertext &cipher);

  void bsgs_linear_transform(
      PhantomCiphertext &rtncipher, PhantomCiphertext &cipher, int totlen, int basicstep, int coeff_logn, const vector<vector<complex<double>>> &fftcoeff);
  void rotated_bsgs_linear_transform(
      PhantomCiphertext &rtncipher, PhantomCiphertext &cipher, int totlen, int basicstep, int coeff_logn, const vector<vector<complex<double>>> &fftcoeff);
  void rotated_nobsgs_linear_transform(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher, int totlen, int coeff_logn, vector<vector<complex<double>>> fftcoeff);

  void bsgs_linear_transform_hoisting(
      PhantomCiphertext &rtncipher, PhantomCiphertext &cipher, int totlen, int basicstep, int coeff_logn, vector<vector<complex<double>>> fftcoeff);
  void rotated_bsgs_linear_transform_hoisting(
      PhantomCiphertext &rtncipher, PhantomCiphertext &cipher, int totlen, int basicstep, int coeff_logn, vector<vector<complex<double>>> fftcoeff);

  void sfl_one_depth(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void sfl_full_one_depth(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void sflinv_one_depth(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void sflinv_one_depth_more_depth(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);

  void sfl(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void sfl_full(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void sflinv(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void sflinv_full(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);

  void sfl_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void sfl_full_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void sfl_half_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void sfl_full_half_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void sflinv_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void sflinv_full_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);

  void sfl_hoisting(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void sfl_full_hoisting(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void sflinv_hoisting(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void sflinv_full_hoisting(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);

  // original bootstrapping
  void coefftoslot(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void slottocoeff(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);

  void coefftoslot_full(PhantomCiphertext &rtncipher1, PhantomCiphertext &rtncipher2, PhantomCiphertext &cipher);
  void slottocoeff_full(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher1, PhantomCiphertext &cipher2);

  // level-3 LT
  void coefftoslot_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void slottocoeff_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void slottocoeff_half_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);

  void coefftoslot_full_3(PhantomCiphertext &rtncipher1, PhantomCiphertext &rtncipher2, PhantomCiphertext &cipher);
  void slottocoeff_full_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher1, PhantomCiphertext &cipher2);
  void slottocoeff_full_half_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher1, PhantomCiphertext &cipher2);
  // original bootstrapping hoisting version
  void coefftoslot_hoisting(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void slottocoeff_hoisting(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);

  void coefftoslot_full_hoisting(PhantomCiphertext &rtncipher1, PhantomCiphertext &rtncipher2, PhantomCiphertext &cipher);
  void slottocoeff_full_hoisting(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher1, PhantomCiphertext &cipher2);

  // one depth bootstrapping
  void coefftoslot_one_depth(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void slottocoeff_one_depth(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);

  void coefftoslot_full_one_depth(PhantomCiphertext &rtncipher1, PhantomCiphertext &rtncipher2, PhantomCiphertext &cipher);
  void slottocoeff_full_one_depth(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher1, PhantomCiphertext &cipher2);

  // mul_first bootstrapping
  void coefftoslot_full_mul_first(PhantomCiphertext &rtncipher1, PhantomCiphertext &rtncipher2, PhantomCiphertext &cipher);
  void modraise_inplace(PhantomCiphertext &cipher);

  // API bootstrapping
  void bootstrap_sparse(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void bootstrap_full(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);

  void bootstrap_sparse_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void bootstrap_full_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);

  void bootstrap_sparse_real_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void bootstrap_full_real_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);

  void bootstrap_sparse_hoisting(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void bootstrap_full_hoisting(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void bootstrap_one_depth(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void bootstrap_more_depth(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void bootstrap(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void bootstrap_inplace(PhantomCiphertext &cipher);

  void bootstrap_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void bootstrap_inplace_3(PhantomCiphertext &cipher);

  void bootstrap_real_3(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void bootstrap_inplace_real_3(PhantomCiphertext &cipher);

  void bootstrap_hoisting(PhantomCiphertext &rtncipher, PhantomCiphertext &cipher);
  void bootstrap_inplace_hoisting(PhantomCiphertext &cipher);
};
