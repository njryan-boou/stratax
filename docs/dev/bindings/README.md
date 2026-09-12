@page dev_bindings Bindings Developer Docs

# Bindings Developer Docs

Python loads the compiled _core extension and re-exports its public symbols from
python/stratax/__init__.py. The C++ core remains the implementation of array
operations; bindings handle Python argument conversion and lifetime.

| Source | Responsibility |
| --- | --- |
| bindings/module.cpp | Metadata, exception registration, and binding order |
| bindings/shape.cpp | Shape constructors, properties, and checked dimension access |
| bindings/vector.cpp, matrix.cpp, tensor.cpp | Owning double constructors, indexing, and operations |
| bindings/boolean_arrays.cpp | Boolean comparison-result classes |
| bindings/views.cpp | Owner-retaining double ArrayView wrapper interface |
| bindings/conversions.cpp, creation.cpp | Module-level conversion/creation overloads |
| bindings/reductions.cpp | Nine reductions and int64 IndexTensor registration |
| bindings/binding_utils | Shared arithmetic, comparison, properties, conversion, and view helpers |

Python integer and tuple access validates before reaching unchecked storage.
Slices retain the source owner and keep integer-selected axes of mixed slice
expressions. The view interface is narrower than C++ ArrayView: no further
slicing, arithmetic, reshape, or reduction bindings.

Argument conversion can raise Python built-in exceptions as well as registered
Stratax exceptions. Scalar helpers reject bool and nonfinite inputs; array
results can still become nonfinite through C++ arithmetic. Keep documentation
and stubs aligned with actual exports, including Boolean and index result types.

Import, source/installed package selection, shape, lifetime, and error contracts
are covered in tests/python. See @ref dev_module and @ref python_api.
