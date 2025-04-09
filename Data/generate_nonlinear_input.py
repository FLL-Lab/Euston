import numpy as np


def gelu(x):
    return x / (1 + np.exp(-1.702*x))


def layernorm(mat, eps=1e-5):
    """实现Layer Normalization（无学习参数版本）"""
    # 计算每行的均值与方差，保持维度以支持广播
    mean = np.mean(mat, axis=1, keepdims=True)
    var = np.var(mat, axis=1, keepdims=True)
    # 归一化
    normalized = (mat - mean) / np.sqrt(var + eps)
    return normalized


def softmax(mat):
    """实现Softmax（带数值稳定优化）"""
    # 每行减去最大值防止溢出
    max_vals = np.max(mat, axis=1, keepdims=True)
    exp_mat = np.exp(mat - max_vals)  # 数值稳定处理
    # 计算每行的和并归一化
    sum_exp = np.sum(exp_mat, axis=1, keepdims=True)
    softmax_mat = exp_mat / sum_exp
    return softmax_mat

np.random.seed(99)

# GELU
interval = [-5, 5]
matrix = np.random.uniform(low=interval[0], high=interval[1], size=(32768, 3072))
np.savetxt(f"input/gelu_input_batched_32768X3072.txt", matrix, fmt="%.4f", delimiter=" ")
matrix = np.loadtxt(f"input/gelu_input_batched_32768X3072.txt")
y = gelu(matrix)
np.savetxt(f"output/gelu_output_batched_32768X3072.txt", y, fmt="%.4f", delimiter=" ")

# LayerNorm
interval = [-2, 2]
matrix = np.random.uniform(low=interval[0], high=interval[1], size=(32768, 768))
np.savetxt(f"input/LayerNorm_input_batched_32768X768.txt", matrix, fmt="%.4f", delimiter=" ")
matrix = np.loadtxt(f"input/LayerNorm_input_batched_32768X768.txt")
y = layernorm(matrix)
np.savetxt(f"output/LayerNorm_output_batched_32768X768.txt", y, fmt="%.4f", delimiter=" ")

# Softmax
interval = [-10, 0]
matrix = np.random.uniform(low=interval[0], high=interval[1], size=(32768, 128))
np.savetxt(f"input/Softmax_input_batched_32768X128.txt", matrix, fmt="%.4f", delimiter=" ")
matrix = np.loadtxt(f"input/Softmax_input_batched_32768X128.txt")
y = softmax(matrix)
np.savetxt(f"output/Softmax_output_batched_32768X128.txt", y, fmt="%.4f", delimiter=" ")
