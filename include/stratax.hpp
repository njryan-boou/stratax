#pragma once

#include <stratax/core/Buffer.hpp>
#include <stratax/core/Concepts.hpp>
#include <stratax/core/Config.hpp>
#include <stratax/core/Exceptions.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Strides.hpp>
#include <stratax/core/Types.hpp>
#include <stratax/core/Validation.hpp>
#include <stratax/core/Slice.hpp>

#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>
#include <stratax/containers/Creation.hpp>
#include <stratax/containers/Conversions.hpp>

#include <stratax/linalg/decomposition.hpp>
#include <stratax/linalg/determinant.hpp>
#include <stratax/linalg/eigen.hpp>
#include <stratax/linalg/inverse.hpp>
#include <stratax/linalg/norms.hpp>
#include <stratax/linalg/solve.hpp>
#include <stratax/linalg/transforms.hpp>

#include <stratax/calculus/curl.hpp>
#include <stratax/calculus/derivative.hpp>
#include <stratax/calculus/divergence.hpp>
#include <stratax/calculus/gradient.hpp>
#include <stratax/calculus/integration.hpp>

#include <stratax/statistics/descriptive.hpp>
#include <stratax/statistics/distributions.hpp>
#include <stratax/statistics/hypothesis.hpp>
#include <stratax/statistics/regression.hpp>

#include <stratax/random/distributions.hpp>
#include <stratax/random/generators.hpp>
#include <stratax/random/sampling.hpp>

#include <stratax/io/binary.hpp>
#include <stratax/io/csv.hpp>
#include <stratax/io/Print.hpp>

#include <stratax/ops/Comparison.hpp>
#include <stratax/ops/Arithmetic.hpp>
#include <stratax/ops/broadcasting.hpp>
#include <stratax/ops/Indexing.hpp>
#include <stratax/ops/logical.hpp>
#include <stratax/ops/Reshape.hpp>
#include <stratax/ops/Slice.hpp>

#include <stratax/detail/kernels.hpp>
#include <stratax/detail/memory.hpp>
#include <stratax/detail/simd.hpp>
#include <stratax/detail/threading.hpp>
#include <stratax/detail/traits.hpp>
#include <stratax/detail/utilities.hpp>
