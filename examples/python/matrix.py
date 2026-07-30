from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from stratax import Matrix, Shape, mean, sum, to_matrix


# Matrices store row-major contiguous values with rank-2 shape metadata.
matrix = Matrix([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])
bias = Matrix(2, 3, 0.5)

# Tuple indexing addresses row and column.
matrix[1, 2] = 9.0

# Arithmetic supports matrix/matrix and matrix/scalar operations.
shifted = matrix + bias
doubled = matrix * 2.0

# Matrix slicing returns a copied matrix.
stepped = matrix[:, ::2]

# Reshape and flatten preserve row-major order.
reshaped = matrix.reshape(Shape([3, 2]))
flattened = matrix.flatten()
roundtrip = to_matrix(reshaped)

# Axis reductions return tensors. keepdims preserves the reduced axis.
row_sums = sum(matrix, 1)
col_sums_keepdims = sum(matrix, 0, keepdims=True)
global_mean = mean(matrix)

print("matrix:")
print(matrix)
print("shifted:")
print(shifted)
print("doubled:")
print(doubled)
print("stepped slice [:, ::2]:")
print(stepped)
print("reshaped tensor:")
print(reshaped)
print("flattened vector:")
print(flattened)
print("roundtrip to matrix:")
print(roundtrip)
print("row sums (axis=1):")
print(row_sums)
print("column sums keepdims (axis=0):")
print(col_sums_keepdims)
print("mean(matrix):", global_mean)
print("rows:", matrix.rows, "cols:", matrix.cols)
print("shape:", matrix.shape)
print("strides:", matrix.strides)
print("value at (1, 2):", matrix[1, 2])
print("flat values:", list(matrix))
