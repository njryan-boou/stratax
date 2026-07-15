from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from stratax import Matrix, Shape, to_matrix



matrix = Matrix([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])
bias = Matrix(2, 3, 0.5)

matrix[1, 2] = 9.0

shifted = matrix + bias
doubled = matrix * 2.0
stepped = matrix[:, ::2]
reshaped = matrix.reshape(Shape([3, 2]))
flattened = matrix.flatten()
roundtrip = to_matrix(reshaped)

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
print("rows:", matrix.rows, "cols:", matrix.cols)
print("value at (1, 2):", matrix[1, 2])
