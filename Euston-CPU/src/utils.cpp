#include "utils.h"

tensor hadamard_product(const tensor& vec1, const tensor& vec2)
{
    if (vec1.size() != vec2.size()) {
        throw invalid_argument("Vectors must be of the same size for Hadamard product.");
    }
    vector<double> result(vec1.size());
	#pragma omp parallel for num_threads(32)
    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] * vec2[i];
    }
    return result;
}
tensor tensor_add_tensor(const tensor& vec1, const tensor& vec2)
{
    if (vec1.size() != vec2.size()) {
        throw invalid_argument("Vectors must be of the same size for element-wise addition.");
    }
    vector<double> result(vec1.size());
	#pragma omp parallel for num_threads(32)
    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] + vec2[i];
    }
    return result;
}
matrix mat_add_tensor(const matrix &A, const tensor &B)
{
    size_t numRows = A.size();
    size_t numCols = A[0].size();
    if (B.size() != numCols) {
        throw out_of_range("Tensor B size does not match the number of columns in matrix A.");
    }
    matrix result(numRows, tensor(numCols, 0.0));
    #pragma omp parallel for num_threads(32)
    for (size_t i = 0; i < numRows; ++i) {
		
        for (size_t j = 0; j < numCols; ++j) {
            result[i][j] = A[i][j] + B[j];
        }
    }
    return result;
}
matrix mat_add_mat(const matrix& A, const matrix& B) 
{
	auto start = chrono::high_resolution_clock::now();
    if (A.size() != B.size() || A.empty() || B.empty() || A[0].size() != B[0].size()) {
        throw invalid_argument("Matrices dimensions must be the same for addition.");
    }

    matrix result(A.size(), tensor(A[0].size(), 0.0));
	#pragma omp parallel for num_threads(32)
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[i].size(); ++j) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }
	auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    // cout << "mat_add_mat Elapsed time: " << elapsed.count() << " s" << endl;
    return result;
}
matrix mat_sub_mat(const matrix& A, const matrix& B) 
{

    if (A.size() != B.size() || A.empty() || B.empty() || A[0].size() != B[0].size()) {
        throw invalid_argument("Matrices dimensions must be the same for addition.");
    }

    matrix result(A.size(), tensor(A[0].size(), 0.0));
	#pragma omp parallel for num_threads(32)
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[i].size(); ++j) {
            result[i][j] = A[i][j] - B[i][j];
        }
    }
    return result;
}

matrix mat_mul_mat(const matrix &A, const matrix &B)
{
    size_t numRowsA = A.size();
    size_t numColsA = A[0].size();
	size_t numRowsB = B.size();
    size_t numColsB = B[0].size();
    
    matrix result(numRowsA, tensor(numColsB, 0.0));
	if(numColsA != numRowsB){
		cout << "A's dimension" << numColsA << " B's dimension" << numRowsB << endl;
		cerr << "Dimensions not match when mat_mul_mat." << endl;
		return result;
	}

   	for (size_t i = 0; i < numRowsA; ++i) {
        for (size_t k = 0; k < numColsA; ++k) {
            double valA = A[i][k];
            for (size_t j = 0; j < numColsB; ++j) {
                result[i][j] += valA * B[k][j];
            }
        }
    }
    return result;
}

void mat_scale(matrix& mat, double scale)
{
	for (size_t i = 0; i < mat.size(); ++i) {
        for (size_t j = 0; j < mat[0].size(); ++j) {
            mat[i][j] *= scale;
        }
    }
}
matrix mat_transpose(const matrix& A)
{
    size_t rows = A.size();
    size_t cols = A.empty() ? 0 : A[0].size();
	matrix result;
    try {
		result.resize(cols);
		
		for(int i = 0; i < cols; i++)
		{
			result[i].resize(rows);
		}
	} catch (const bad_alloc& e) {
		cerr << "Memory allocation failed: " << e.what() << endl;
		throw;
	}

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[j][i] = A[i][j];
        }
    }
    return result;
}

matrix readMatrix(const string &filename, int rows, int cols) 
{
  matrix mat(rows, tensor(cols));
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

// 将 matrix 转换为 Eigen::MatrixXd
Eigen::MatrixXd vectorToEigen(const matrix& mat) {
    int rows = mat.size();
    int cols = mat[0].size();
    Eigen::MatrixXd eigenMat(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            eigenMat(i, j) = mat[i][j];
        }
    }
    return eigenMat;
}

matrix eigenToVector(const Eigen::MatrixXd& eigenMat) {
    int rows = eigenMat.rows();
    int cols = eigenMat.cols();
    matrix mat(rows, tensor(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat[i][j] = eigenMat(i, j);
        }
    }
    return mat;
}

void svd_decomposition(const matrix& input, matrix& U, tensor& D, matrix& V) {
    Eigen::MatrixXd eigenMat = vectorToEigen(input);

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(eigenMat, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd eigenU = svd.matrixU();
    Eigen::MatrixXd eigenV = svd.matrixV();
    Eigen::VectorXd eigenD = svd.singularValues();

    U = eigenToVector(eigenU);
    V = eigenToVector(eigenV);
	V = mat_transpose(V);
    D = tensor(eigenD.data(), eigenD.data() + eigenD.size());
}

bool verify_svd(const matrix& original, const matrix& U, const tensor& S, const matrix& V) {
    Eigen::MatrixXd eigenU = vectorToEigen(U);
    Eigen::MatrixXd eigenV = vectorToEigen(V);
    Eigen::VectorXd eigenS = Eigen::Map<const Eigen::VectorXd>(S.data(), S.size());
    
    Eigen::MatrixXd sigma = Eigen::MatrixXd::Zero(eigenS.size(), eigenS.size());
    for (int i = 0; i < eigenS.size(); ++i) {
        sigma(i, i) = eigenS(i);
    }
    
    Eigen::MatrixXd reconstructed = eigenU * sigma * eigenV.transpose();
    matrix reconstructedVec = eigenToVector(reconstructed);

    const double tolerance = 1e-6; 
    for (size_t i = 0; i < original.size(); ++i) {
        for (size_t j = 0; j < original[0].size(); ++j) {
            if (abs(original[i][j] - reconstructedVec[i][j]) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

void save_matrix(const matrix& cpp_vector, string filename)
{
	ofstream file(filename);
	 if (!file.is_open()) {
        cerr << "Error opening file for writing: " << filename << endl;
        return;
    }
    for (const auto& row : cpp_vector) {
        for (const auto& val : row) {
            file << fixed << setprecision(6) << val << " ";
        }
        file << "\n";
    }
    file.close();
}
matrix load_matrix(const string& filename) 
{
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file for reading: " << filename << endl;
        return matrix();
    }

    matrix mat;
    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        vector<double> row;
        double val;
        while (iss >> val) {
            row.push_back(val);
        }
        mat.push_back(row);
    }

    file.close();
    return mat;
}

void print_error(string message)
{
	cerr << "Error: " + message << endl;
}

void print_tensor(tensor& t, string des, int num)
{
	cout << des;
	for(int i = 0; i < num; i++)
	{
		cout << " " << t[i] ;
	}
	cout << endl;
}

void Rectime::start() {
	start_time = chrono::high_resolution_clock::now();
	running = true;
}

double Rectime::end() {
	if (running) {
		end_time = chrono::high_resolution_clock::now();
		running = false;
	}
	return chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
}

double Rectime::elapsed(string operation) {
	chrono::time_point<chrono::high_resolution_clock> end;
	if (running) {
		end = chrono::high_resolution_clock::now();
	} else {
		end = end_time;
	}
	auto elapsed_time = chrono::duration<double>(end - start_time).count();
	cout << operation + " Elapsed time: " << fixed << setprecision(6) << elapsed_time << " seconds\n";
	start();
	return elapsed_time;
}

void plain_rotate(tensor& vec, int positions) {
    if (vec.empty()) {
        return;
    }
    int n = vec.size();
    positions = ((positions % n) + n) % n;

    if (positions > 0) {
        rotate(vec.begin(), vec.begin() + positions, vec.end());
    } else if (positions < 0) {
        rotate(vec.rbegin(), vec.rbegin() - positions, vec.rend());
    }
}