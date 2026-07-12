from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from stratax import Vector


values = Vector([1.0, 2.0, 3.0])
weights = Vector(3, 10.0)

values[1] = 5.0

sum_values = values + weights
scaled = values * 2.0

print("values:", values)
print("sum:", sum_values)
print("scaled:", scaled)
print("first value:", values[0])
print("shape:", values.shape)
