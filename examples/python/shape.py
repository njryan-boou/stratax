from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from stratax import Shape


image_shape = Shape([3, 224, 224])
empty_shape = Shape()
zero_sized = Shape([4, 0, 2])

print("image shape:", image_shape)
print("rank:", image_shape.rank)
print("elements:", image_shape.elements)
print("first dimension:", image_shape[0])

print("empty shape:", empty_shape)
print("zero-sized shape:", zero_sized, "has", zero_sized.elements, "elements")
