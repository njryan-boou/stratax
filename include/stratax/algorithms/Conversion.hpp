#pragma once

#include <algorithm>
#include <cstddef>

#include <stratax/containers/Matrix.hpp>
#include <stratax/containers/Tensor.hpp>
#include <stratax/containers/Vector.hpp>
#include <stratax/concepts/Numeric.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/validation/Validation.hpp>

namespace stratax::conversion {

namespace detail {

/**
 * @brief Determines whether a shape can be represented by a Vector.
 *
 * Every rank-one shape is accepted. Higher-rank shapes are accepted only when
 * exactly one dimension is greater than one; dimensions of size zero or one do
 * not count as non-singleton dimensions.
 *
 * @param shape Shape to inspect.
 * @return `true` when @p shape is vector-compatible; otherwise `false`.
 * @complexity O(shape.rank()).
 */
inline bool is_vector_shape(const stratax::core::Shape& shape)
{
	if (shape.rank() == 1)
	{
		return true;
	}
    
	std::size_t non_singleton = 0;

	for (std::size_t dim : shape)
	{
		if (dim > 1)
		{
			++non_singleton;
		}
	}

	return non_singleton == 1;
}

/**
 * @brief Determines whether a shape can be represented by a Matrix.
 *
 * Every rank-two shape is accepted. Shapes of any other rank are accepted only
 * when exactly two dimensions are greater than one; dimensions of size zero or
 * one do not count as non-singleton dimensions.
 *
 * @param shape Shape to inspect.
 * @return `true` when @p shape is matrix-compatible; otherwise `false`.
 * @complexity O(shape.rank()).
 */
inline bool is_matrix_shape(const stratax::core::Shape& shape)
{
	if (shape.rank() == 2)
	{
		return true;
	}

	std::size_t non_singleton = 0;

	for (std::size_t dim : shape)
	{
		if (dim > 1)
		{
			++non_singleton;
		}
	}

	return non_singleton == 2;
}

/**
 * @brief Produces the rank-two shape used by a matrix conversion.
 *
 * Rank-two shapes are returned unchanged. For other ranks, dimensions greater
 * than one are retained in their original order and all zero or singleton
 * dimensions are removed. Callers must first establish matrix compatibility
 * with is_matrix_shape().
 *
 * @param shape Matrix-compatible source shape.
 * @return The two non-singleton dimensions, or @p shape when it is rank two.
 * @throws std::bad_alloc If storage for the filtered dimensions cannot be allocated.
 * @complexity O(shape.rank()).
 */
inline stratax::core::Shape matrix_shape(const stratax::core::Shape& shape)
{
	if (shape.rank() == 2)
	{
		return shape;
	}

	std::vector<std::size_t> dims;
	dims.reserve(shape.rank());

	for (std::size_t dim : shape)
	{
		if (dim > 1)
		{
			dims.push_back(dim);
		}
	}

	return stratax::core::Shape(dims);
}

} // namespace detail

/**
 * @brief Copies a vector-compatible Stratax array into a Vector.
 *
 * The result contains the source elements in flat row-major iterator order and
 * owns storage independent of @p arr. Rank-one sources are always compatible;
 * other ranks must contain exactly one dimension greater than one.
 *
 * @tparam A Vector, Matrix, or Tensor type satisfying Array.
 * @param arr Source array to convert.
 * @return Owning Vector with `A::value_type` and `arr.size()` elements.
 * @throws Exceptions::ShapeError If the source shape is not vector-compatible.
 * @throws std::bad_alloc If result allocation fails.
 * @complexity O(arr.size() + arr.rank()).
 */
template<Array A>
[[nodiscard]]
stratax::container::Vector<typename A::value_type>
to_vector(const A& arr)
{
	if (!detail::is_vector_shape(arr.shape()))
	{
		throw Exceptions::ShapeError(
			"Array cannot be converted to a Vector."
		);
	}

	stratax::container::Vector<typename A::value_type> result(arr.size());
	std::copy(arr.begin(), arr.end(), result.begin());

	return result;
}

/**
 * @brief Copies a matrix-compatible Stratax array into a Matrix.
 *
 * Rank-two sources retain their shape. For other ranks, zero and singleton
 * dimensions are removed and the two remaining non-singleton dimensions form
 * the result shape. Values retain flat row-major iterator order, and the result
 * owns independent storage.
 *
 * @tparam A Vector, Matrix, or Tensor type satisfying Array.
 * @param arr Source array to convert.
 * @return Owning Matrix with `A::value_type` and the compatible matrix shape.
 * @throws Exceptions::ShapeError If the source shape is not matrix-compatible.
 * @throws Exceptions::DimensionError If result shape arithmetic overflows.
 * @throws std::bad_alloc If result or shape allocation fails.
 * @complexity O(arr.size() + arr.rank()).
 */
template<Array A>
[[nodiscard]]
stratax::container::Matrix<typename A::value_type>
to_matrix(const A& arr)
{
	if (!detail::is_matrix_shape(arr.shape()))
	{
		throw Exceptions::ShapeError(
			"Array cannot be converted to a Matrix.");
	}

	const auto shape = detail::matrix_shape(arr.shape());
	stratax::container::Matrix<typename A::value_type> result(shape);
	std::copy(arr.begin(), arr.end(), result.begin());

	return result;
}

/**
 * @brief Copies a Stratax array into a Tensor of the same shape.
 *
 * Values retain flat row-major iterator order. The returned tensor preserves
 * the source shape exactly and owns storage independent of @p arr.
 *
 * @tparam A Vector, Matrix, or Tensor type satisfying Array.
 * @param arr Source array to convert.
 * @return Owning Tensor with `A::value_type`, the same shape, and copied values.
 * @throws Exceptions::DimensionError If result stride arithmetic overflows.
 * @throws std::bad_alloc If result allocation fails.
 * @complexity O(arr.size() + arr.rank()).
 */
template<Array A>
[[nodiscard]]
stratax::container::Tensor<typename A::value_type>
to_tensor(const A& arr)
{
	stratax::container::Tensor<typename A::value_type> result(arr.shape());
	std::copy(arr.begin(), arr.end(), result.begin());

	return result;
}

/**
 * @brief Casts every element of a Vector to another numeric type.
 *
 * Each value is converted with `static_cast<To>`. Consequently, the usual C++
 * narrowing, truncation, sign conversion, and precision-loss rules apply. The
 * result preserves the source shape and owns independent storage.
 *
 * @tparam To Destination numeric element type.
 * @tparam From Source numeric element type.
 * @param vec Vector whose elements are converted.
 * @return Owning Vector of `To` containing the converted values.
 * @throws std::bad_alloc If result allocation fails.
 * @complexity O(vec.size()).
 */
template<typename To, typename From>
requires Numeric<To> && Numeric<From>
stratax::container::Vector<To>
astype(const stratax::container::Vector<From>& vec)
{
	stratax::container::Vector<To> result(vec.shape());

	for (std::size_t i = 0; i < vec.size(); ++i)
	{
		result[i] = static_cast<To>(vec[i]);
	}

	return result;
}

/**
 * @brief Casts every element of a Matrix to another numeric type.
 *
 * Each value is converted with `static_cast<To>`, using the standard C++
 * numeric conversion rules. The result preserves the source shape and flat
 * row-major order and owns independent storage.
 *
 * @tparam To Destination numeric element type.
 * @tparam From Source numeric element type.
 * @param mat Matrix whose elements are converted.
 * @return Owning Matrix of `To` containing the converted values.
 * @throws Exceptions::DimensionError If result shape arithmetic overflows.
 * @throws std::bad_alloc If result allocation fails.
 * @complexity O(mat.size()).
 */
template<typename To, typename From>
requires Numeric<To> && Numeric<From>
stratax::container::Matrix<To>
astype(const stratax::container::Matrix<From>& mat)
{
	stratax::container::Matrix<To> result(mat.shape());

	for (std::size_t i = 0; i < mat.size(); ++i)
	{
		result[i] = static_cast<To>(mat[i]);
	}

	return result;
}

/**
 * @brief Casts every element of a Tensor to another numeric type.
 *
 * Each value is converted with `static_cast<To>`, using the standard C++
 * numeric conversion rules. The result preserves the complete source shape
 * and flat row-major order and owns independent storage.
 *
 * @tparam To Destination numeric element type.
 * @tparam From Source numeric element type.
 * @param tensor Tensor whose elements are converted.
 * @return Owning Tensor of `To` containing the converted values.
 * @throws Exceptions::DimensionError If result stride arithmetic overflows.
 * @throws std::bad_alloc If result allocation fails.
 * @complexity O(tensor.size()).
 */
template<typename To, typename From>
requires Numeric<To> && Numeric<From>
stratax::container::Tensor<To>
astype(const stratax::container::Tensor<From>& tensor)
{
	stratax::container::Tensor<To> result(tensor.shape());

	for (std::size_t i = 0; i < tensor.size(); ++i)
	{
		result[i] = static_cast<To>(tensor[i]);
	}

	return result;
}

} // namespace stratax::conversion
