import numpy as np

def diagpack_matrix(input_matrix):
    # 获取输入矩阵的行数和列数
    rows = len(input_matrix)
    cols = len(input_matrix[0]) if rows > 0 else 0
    
    # 初始化输出矩阵（cols行，每行包含rows个元素）
    output = []
    for j in range(cols):
        temp = []
        for i in range(rows):
            # 计算环形偏移后的列索引 (i+j) mod cols
            col_idx = (i + j) % cols
            # 从输入矩阵中提取元素
            temp.append(input_matrix[i][col_idx])
        output.append(temp)
    return np.array(output)

np.random.seed(99)

# Matrix
# IWMM
interval = [-2, 2]
matrix_input = np.random.uniform(low=interval[0], high=interval[1], size=(4096, 4096))
matrix_weight = np.random.uniform(low=interval[0], high=interval[1], size=(4096, 128))
np.savetxt(f"input/IWMM_LLAMA_input_32x128X4096.txt", matrix_input, fmt="%.4f", delimiter=" ")
np.savetxt(f"input/IWMM_LLAMA_weight_4096x128.txt", matrix_weight, fmt="%.4f", delimiter=" ")
matrix_input = np.loadtxt(f"input/IWMM_LLAMA_input_32x128X4096.txt")
matrix_weight = np.loadtxt(f"input/IWMM_LLAMA_weight_4096x128.txt")
matrix_output = matrix_input @ matrix_weight
np.savetxt(f"output/IWMM_LLAMA_output_32x128X128.txt", matrix_output, fmt="%.4f", delimiter=" ")


# CRMM: Column-packed CipherMatrix X Row-packed CipherMatrix
matrix_input = []
matrix_output = []
for i in range(32):
    mat_i = np.random.uniform(low=interval[0], high=interval[1], size=(128, 128))
    np.savetxt(f"input/tmpmat.txt", mat_i, fmt="%.4f", delimiter=" ")
    mat_i = np.loadtxt(f"input/tmpmat.txt")
    matrix_input.append(mat_i)
    mat = diagpack_matrix(mat_i @ mat_i.T)
    matrix_output.append(mat.T)
matrix_input = np.array(matrix_input)
matrix_input = np.concatenate(matrix_input, axis=0)
matrix_output = np.array(matrix_output)
matrix_output = np.concatenate(matrix_output, axis=0)
np.savetxt(f"input/CCMM_LLAMA_input_32x128X128.txt", matrix_input, fmt="%.4f", delimiter=" ")
np.savetxt(f"output/CCMM_LLAMA_output_32x128X128_diagpack.txt", matrix_output, fmt="%.4f", delimiter=" ")

# DCMM: Diagnal-packed CipherMatrix X Column-packed CipherMatrix
left_mat_input = []
right_mat_input = []
mat_output = []
for _ in range(32):
    leftmat_i = np.random.uniform(low=interval[0], high=interval[1], size=(128, 128))
    np.savetxt(f"input/tmpmat.txt", leftmat_i, fmt="%.4f", delimiter=" ")
    leftmat_i = np.loadtxt(f"input/tmpmat.txt")
    left_mat_input.append(diagpack_matrix(leftmat_i).T)
    rightmat_i = np.random.uniform(low=interval[0], high=interval[1], size=(128, 128))
    np.savetxt(f"input/tmpmat.txt", rightmat_i, fmt="%.4f", delimiter=" ")
    rightmat_i = np.loadtxt(f"input/tmpmat.txt")
    right_mat_input.append(rightmat_i)
    mat = leftmat_i @ rightmat_i
    mat_output.append(mat)
left_mat_input = np.array(left_mat_input)
left_mat_input = np.concatenate(left_mat_input, axis=0)
right_mat_input = np.array(right_mat_input)
right_mat_input = np.concatenate(right_mat_input, axis=0)
mat_output = np.array(mat_output)
mat_output = np.concatenate(mat_output, axis=0)
np.savetxt(f"input/DCMM_LLAMA_left_input_32x128X128.txt", left_mat_input, fmt="%.4f", delimiter=" ")
np.savetxt(f"input/DCMM_LLAMA_right_input_32x128X128.txt", right_mat_input, fmt="%.4f", delimiter=" ")
np.savetxt(f"output/DCMM_LLAMA_output_32x128X128.txt", mat_output, fmt="%.4f", delimiter=" ")

# CPMM-1
left_mat_input = np.random.uniform(low=interval[0], high=interval[1], size=(4096, 4096))
np.savetxt(f"input/tmpmat.txt", left_mat_input, fmt="%.4f", delimiter=" ")
left_mat_input = np.loadtxt(f"input/tmpmat.txt")
right_mat_input = np.random.uniform(low=interval[0], high=interval[1], size=(4096, 64)) # 14336=64*224
np.savetxt(f"input/tmpmat.txt", right_mat_input, fmt="%.4f", delimiter=" ")
right_mat_input = np.loadtxt(f"input/tmpmat.txt")
mat_output = left_mat_input @ right_mat_input
np.savetxt(f"input/CPMM1_LLAMA_left_input_32x128X4096.txt", left_mat_input, fmt="%.4f", delimiter=" ")
np.savetxt(f"input/CPMM1_LLAMA_right_input_4096X64.txt", right_mat_input, fmt="%.4f", delimiter=" ")
np.savetxt(f"output/CPMM1_LLAMA_output_32x128X64.txt", mat_output, fmt="%.4f", delimiter=" ")

# CPMM-2
left_mat_input = np.random.uniform(low=interval[0], high=interval[1], size=(4096, 14336))
np.savetxt(f"input/tmpmat.txt", left_mat_input, fmt="%.4f", delimiter=" ")
left_mat_input = np.loadtxt(f"input/tmpmat.txt")
right_mat_input = np.random.uniform(low=interval[0], high=interval[1], size=(14336, 128)) # 4096=128*32
np.savetxt(f"input/tmpmat.txt", right_mat_input, fmt="%.4f", delimiter=" ")
right_mat_input = np.loadtxt(f"input/tmpmat.txt")
mat_output = left_mat_input @ right_mat_input
np.savetxt(f"input/CPMM2_LLAMA_left_input_32x128X14336.txt", left_mat_input, fmt="%.4f", delimiter=" ")
np.savetxt(f"input/CPMM2_LLAMA_right_input_14336X64.txt", right_mat_input, fmt="%.4f", delimiter=" ")
np.savetxt(f"output/CPMM2_LLAMA_output_32x128X64.txt", mat_output, fmt="%.4f", delimiter=" ")