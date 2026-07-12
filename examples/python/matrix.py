from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from stratax import Matrix



matrix = Matrix([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
])
bias = Matrix(2, 3, 0.5)

matrix[1, 2] = 9.0

shifted = matrix + bias
doubled = matrix * 2.0

print("matrix:")
print(matrix)
print("shifted:")
print(shifted)
print("doubled:")
print(doubled)
print("rows:", matrix.rows, "cols:", matrix.cols)
print("value at (1, 2):", matrix[1, 2])
