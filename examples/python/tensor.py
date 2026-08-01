from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from stratax import Shape, Tensor, identity, mean, sum, to_matrix


# Tensors store flat row-major values behind N-dimensional shape metadata.
tensor = Tensor(Shape([2, 2, 2]), 1.0)
eye = identity(3)

# Tuple indexing follows the tensor rank.
tensor[0, 0, 0] = 10.0
tensor[1, 1, 1] = 20.0

# Tensor arithmetic supports scalars and matching tensor shapes.
shifted = tensor + 2.0
negated = -tensor

# Slicing can mix full-axis slices and stepped slices.
stepped = tensor[:, :, ::2]

# Reshape, flatten, and conversion helpers preserve row-major values.
reshaped = tensor.reshape(Shape([4, 2]))
flattened = tensor.flatten()
as_matrix = to_matrix(eye)

# Reductions can target negative axes and can keep reduced dimensions.
sum_all = sum(tensor)
sum_last_axis = sum(tensor, -1)
mean_keepdims = mean(tensor, 0, keepdims=True)

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
print("sum(tensor):", sum_all)
print("sum over last axis (axis=-1):")
print(sum_last_axis)
print("mean keepdims over axis 0:")
print(mean_keepdims)
print("rank:", tensor.rank)
print("shape:", tensor.shape)
print("strides:", tensor.strides)
print("linear index 7:", tensor[7])
print("flat values:", tensor.tolist())
