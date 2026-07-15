from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from stratax import Shape, Tensor, identity, to_matrix


tensor = Tensor(Shape([2, 2, 2]), 1.0)
eye = identity(3)

tensor[0, 0, 0] = 10.0
tensor[1, 1, 1] = 20.0

shifted = tensor + 2.0
negated = -tensor
stepped = tensor[:, :, ::2]
reshaped = tensor.reshape(Shape([4, 2]))
flattened = tensor.flatten()
as_matrix = to_matrix(eye)

print("tensor:")
print(tensor)
print("shifted:")
print(shifted)
print("negated:")
print(negated)
print("stepped slice [:, :, ::2]:")
print(stepped)
print("reshaped tensor:")
print(reshaped)
print("flattened vector:")
print(flattened)
print("identity via creation helper:")
print(eye)
print("identity converted to matrix:")
print(as_matrix)
print("rank:", tensor.rank)
print("shape:", tensor.shape)
print("linear index 7:", tensor[7])
