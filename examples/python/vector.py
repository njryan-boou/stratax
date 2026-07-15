from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from stratax import Shape, Vector, to_tensor, to_vector


values = Vector([1.0, 2.0, 3.0, 4.0, 5.0])
weights = Vector(values.size, 10.0)

values[1] = 5.0

sum_values = values + weights
scaled = values * 2.0
stepped = values[::2]
reshaped = values.reshape(Shape([1, values.size]))
as_tensor = to_tensor(values)
roundtrip = to_vector(as_tensor)

print("values:", values)
print("sum:", sum_values)
print("scaled:", scaled)
print("stepped slice [::2]:", stepped)
print("reshaped tensor:", reshaped)
print("converted to tensor:", as_tensor)
print("roundtrip to vector:", roundtrip)
print("first value:", values[0])
print("shape:", values.shape)
