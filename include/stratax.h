#pragma once

#include <stratax/core/containers/Buffer.hpp>
#include <stratax/core/Concepts.hpp>
#include <stratax/core/Config.hpp>
#include <stratax/core/Exceptions.hpp>
#include <stratax/core/containers/Shape.hpp>
#include <stratax/core/containers/Strides.hpp>
#include <stratax/core/Types.hpp>
#include <stratax/core/validation/Validation.hpp>
#include <stratax/core/Slice.hpp>

#include <stratax/core/containers/Matrix.hpp>
#include <stratax/core/containers/Tensor.hpp>
#include <stratax/core/containers/Vector.hpp>

#include <stratax/core/algorithms/Creation.hpp>
#include <stratax/core/algorithms/Conversions.hpp>
#include <stratax/core/algorithms/Reshape.hpp>
#include <stratax/core/algorithms/Reductions.hpp>

#include <stratax/core/io/Print.hpp>

#include <stratax/core/ops/Comparison.hpp>
#include <stratax/core/ops/Arithmetic.hpp>
#include <stratax/core/ops/Bitwise.hpp>
#include <stratax/core/ops/Broadcasting.hpp>
#include <stratax/core/ops/Indexing.hpp>
#include <stratax/core/ops/Slice.hpp>

namespace stratax {

// Public array types.
using core::Shape;
using core::Slice;
using core::Strides;
using container::Matrix;
using container::Tensor;
using container::Vector;

// Common public algorithms.
using ::astype;
using ::flatten;
using ::reshape;
using ::slice;
using ::to_matrix;
using ::to_tensor;
using ::to_vector;
using ::creation::full;
using ::creation::identity;
using ::creation::ones;
using ::creation::zeros;
using ::reduction::argmax;
using ::reduction::argmin;
using ::reduction::max;
using ::reduction::mean;
using ::reduction::min;
using ::reduction::prod;
using ::reduction::std;
using ::reduction::sum;
using ::reduction::var;

namespace arrays {
    using ::stratax::Matrix;
    using ::stratax::Shape;
    using ::stratax::Slice;
    using ::stratax::Strides;
    using ::stratax::Tensor;
    using ::stratax::Vector;
}

namespace creation {
    using ::creation::full;
    using ::creation::identity;
    using ::creation::ones;
    using ::creation::zeros;
}

namespace conversions {
    using ::astype;
    using ::to_matrix;
    using ::to_tensor;
    using ::to_vector;
}

namespace reductions {
    using ::reduction::argmax;
    using ::reduction::argmin;
    using ::reduction::max;
    using ::reduction::mean;
    using ::reduction::min;
    using ::reduction::prod;
    using ::reduction::std;
    using ::reduction::sum;
    using ::reduction::var;
}

namespace transforms {
    using ::flatten;
    using ::reshape;
}

namespace slicing {
    using ::slice;
}

namespace errors = ::Exceptions;

using ::Exceptions::AxisError;
using ::Exceptions::BroadcastError;
using ::Exceptions::DimensionError;
using ::Exceptions::IndexError;
using ::Exceptions::ShapeError;
using ::Exceptions::StrataxError;
using ::Exceptions::TypeError;
using ::Exceptions::ZeroDivisionError;

} // namespace stratax

namespace stratax {

// Public array types.
using core::Shape;
using core::Slice;
using core::Strides;
using container::Matrix;
using container::Tensor;
using container::Vector;

// Common public algorithms.
using ::astype;
using ::flatten;
using ::reshape;
using ::slice;
using ::broadcasted_op;
using ::broadcasted_shape;
using ::validate_broadcast;
using ::to_matrix;
using ::to_tensor;
using ::to_vector;
using ::creation::full;
using ::creation::identity;
using ::creation::ones;
using ::creation::zeros;
using ::reduction::argmax;
using ::reduction::argmin;
using ::reduction::max;
using ::reduction::mean;
using ::reduction::min;
using ::reduction::prod;
using ::reduction::std;
using ::reduction::sum;
using ::reduction::var;

namespace arrays {
    using ::stratax::Matrix;
    using ::stratax::Shape;
    using ::stratax::Slice;
    using ::stratax::Strides;
    using ::stratax::Tensor;
    using ::stratax::Vector;
}

namespace creation {
    using ::creation::full;
    using ::creation::identity;
    using ::creation::ones;
    using ::creation::zeros;
}

namespace conversions {
    using ::astype;
    using ::to_matrix;
    using ::to_tensor;
    using ::to_vector;
}

namespace reductions {
    using ::reduction::argmax;
    using ::reduction::argmin;
    using ::reduction::max;
    using ::reduction::mean;
    using ::reduction::min;
    using ::reduction::prod;
    using ::reduction::std;
    using ::reduction::sum;
    using ::reduction::var;
}

namespace transforms {
    using ::flatten;
    using ::reshape;
}

namespace slicing {
    using ::slice;
}

namespace broadcasting {
    using ::broadcasted_op;
    using ::broadcasted_shape;
    using ::validate_broadcast;
}

namespace errors = ::Exceptions;

using ::Exceptions::AxisError;
using ::Exceptions::BroadcastError;
using ::Exceptions::DimensionError;
using ::Exceptions::IndexError;
using ::Exceptions::ShapeError;
using ::Exceptions::StrataxError;
using ::Exceptions::TypeError;
using ::Exceptions::ZeroDivisionError;

} // namespace stratax
