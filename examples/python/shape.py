from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from stratax import Shape


# Shape stores dimensions only. It does not own array values.
image_shape = Shape([3, 224, 224])
empty_shape = Shape()
zero_sized = Shape([4, 0, 2])
copied = Shape(image_shape)

print("image shape:", image_shape)
print("rank:", image_shape.rank)
print("elements:", image_shape.elements)
print("first dimension:", image_shape[0])
print("last dimension:", image_shape[-1])
print("dimensions:", list(image_shape))
print("copied equals original:", copied == image_shape)

print("empty shape:", empty_shape)
print("zero-sized shape:", zero_sized, "has", zero_sized.elements, "elements")
