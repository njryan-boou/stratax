#pragma once

#include <stratax/core/ArrayBase.hpp>
#include <stratax/core/Buffer.hpp>
#include <stratax/concepts/Numeric.hpp>
#include <stratax/core/Config.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/Strides.hpp>
#include <stratax/core/Types.hpp>
#include <stratax/core/DTypeTraits.hpp>
#include <stratax/core/Promotion.hpp>
#include <stratax/core/validation/Validation.hpp>
#include <stratax/core/Slice.hpp>

#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>

#include <stratax/algorithms/Creation.hpp>
#include <stratax/algorithms/Conversion.hpp>
#include <stratax/algorithms/Reshape.hpp>
#include <stratax/algorithms/Reductions.hpp>

#include <stratax/io/Print.hpp>

#include <stratax/ops/Comparison.hpp>
#include <stratax/ops/Arithmetic.hpp>
#include <stratax/ops/Bitwise.hpp>
#include <stratax/ops/Broadcasting.hpp>
#include <stratax/indexing/Indexing.hpp>
#include <stratax/indexing/Slicing.hpp>
