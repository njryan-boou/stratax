from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from stratax import Shape, Tensor


tensor = Tensor(Shape([2, 2, 2]), 1.0)

tensor[0, 0, 0] = 10.0
tensor[1, 1, 1] = 20.0

shifted = tensor + 2.0
negated = -tensor

print("tensor:")
print(tensor)
print("shifted:")
print(shifted)
print("negated:")
print(negated)
print("rank:", tensor.rank)
print("shape:", tensor.shape)
print("linear index 7:", tensor[7])
