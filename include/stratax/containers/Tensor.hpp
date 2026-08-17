// TODO: make normalize flat offset more explicit

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <vector>

#include <stratax/concepts/Numeric.hpp>
#include <stratax/core/ArrayBase.hpp>
#include <stratax/core/Shape.hpp>

namespace stratax::container {

/**
 * @brief Arbitrary-rank owning array of numeric values.
 *
 * Tensor combines contiguous element storage with a logical Shape and its
 * canonical row-major Strides. It supports unchecked multidimensional access
 * with already-normalized unsigned indices and checked multidimensional access
 * with signed, Python-style indices. The flat container interface is inherited
 * from core::ArrayBase.
 *
 * @tparam T Numeric element type satisfying the Numeric concept.
 *
 * @par Invariants
 * - `size() == shape().elements()`.
 * - `rank() == shape().rank() == strides().rank()`.
 * - Elements are contiguous in row-major order.
 *
 * @par Example
 * @code
 * stratax::container::Tensor<double> tensor(
 *     stratax::core::Shape{2, 3, 4},
 *     0.0);
 *
 * tensor(1, 2, 3) = 7.0; // unchecked multidimensional access
 * tensor.at(-1, -1, -1); // 7.0; checked negative indices
 * tensor.at(-1);         // checked flat access inherited from ArrayBase
 * @endcode
 *
 * @see core::ArrayBase
 * @see core::Shape
 * @see core::Strides
 */
template<typename T>
requires Numeric<T>
class Tensor : public core::ArrayBase<T>
{
public:
	/** @brief Stored element type inherited from ArrayBase. */
	using value_type = typename core::ArrayBase<T>::value_type;
	/** @brief Unsigned type used for element counts and normalized indices. */
	using size_type = typename core::ArrayBase<T>::size_type;
	/** @brief Signed type used for checked indices and iterator distances. */
	using difference_type = typename core::ArrayBase<T>::difference_type;
	/** @brief Mutable element reference type. */
	using reference = typename core::ArrayBase<T>::reference;
	/** @brief Read-only element reference type. */
	using const_reference = typename core::ArrayBase<T>::const_reference;
	/** @brief Mutable element pointer type. */
	using pointer = typename core::ArrayBase<T>::pointer;
	/** @brief Read-only element pointer type. */
	using const_pointer = typename core::ArrayBase<T>::const_pointer;
	/** @brief Mutable contiguous random-access iterator type. */
	using iterator = typename core::ArrayBase<T>::iterator;
	/** @brief Read-only contiguous random-access iterator type. */
	using const_iterator = typename core::ArrayBase<T>::const_iterator;
	/** @brief Mutable reverse iterator type. */
	using reverse_iterator = typename core::ArrayBase<T>::reverse_iterator;
	/** @brief Read-only reverse iterator type. */
	using const_reverse_iterator = typename core::ArrayBase<T>::const_reverse_iterator;

protected:
	/** @brief Exposes ArrayBase's unchecked multidimensional offset helper. */
	using core::ArrayBase<T>::flat_offset;
	/** @brief Exposes ArrayBase's checked multidimensional offset helper. */
	using core::ArrayBase<T>::normalized_flat_offset;

public:
	/** @brief Retains the inherited checked flat `at(index)` overloads. */
	using core::ArrayBase<T>::at;

	/**
	 * @brief Constructs an empty rank-one tensor with shape `{0}`.
	 * @complexity O(1).
	 */
	Tensor() : Tensor(core::Shape{0}) {}

	/**
	 * @brief Constructs value-initialized storage for an arbitrary shape.
	 * @param shape Logical dimensions in outermost-to-innermost order.
	 * @throws Exceptions::DimensionError If the element or stride count overflows.
	 * @throws std::bad_alloc If storage allocation fails.
	 * @throws Any exception thrown while value-initializing a value_type.
	 * @complexity O(shape.elements() + shape.rank()).
	 */
	explicit Tensor(const core::Shape& shape)
		: core::ArrayBase<T>(shape)
	{}

	/**
	 * @brief Constructs a tensor filled with copies of @p value.
	 * @param shape Logical dimensions in outermost-to-innermost order.
	 * @param value Value copied into every element.
	 * @throws Exceptions::DimensionError If the element or stride count overflows.
	 * @throws std::bad_alloc If storage allocation fails.
	 * @throws Any exception thrown by value_type's copy constructor.
	 * @complexity O(shape.elements() + shape.rank()).
	 */
	Tensor(const core::Shape& shape, const_reference value)
		: core::ArrayBase<T>(shape, value)
	{}

	/**
	 * @brief Returns an element using unchecked variadic indices.
	 *
	 * @tparam Rest Integral types of the remaining index components.
	 * @param first First normalized index component.
	 * @param rest Remaining normalized index components.
	 * @pre Exactly `rank()` components are supplied and every component is
	 *      non-negative and within its corresponding dimension.
	 * @complexity O(rank()).
	 */
	template<typename... Rest>
	requires ((std::is_integral_v<Rest>) && ...)
	reference operator()(size_type first, Rest... rest)
	{
		std::array<size_type, sizeof...(Rest) + 1> indices{
			first,
			static_cast<size_type>(rest)...
		};

		return (*this)[flat_offset(indices)];
	}

	/**
	 * @brief Returns an element using unchecked variadic indices.
	 *
	 * @tparam Rest Integral types of the remaining index components.
	 * @param first First normalized index component.
	 * @param rest Remaining normalized index components.
	 * @pre Exactly `rank()` components are supplied and every component is
	 *      non-negative and within its corresponding dimension.
	 * @complexity O(rank()).
	 */
	template<typename... Rest>
	requires ((std::is_integral_v<Rest>) && ...)
	const_reference operator()(size_type first, Rest... rest) const
	{
		std::array<size_type, sizeof...(Rest) + 1> indices{
			first,
			static_cast<size_type>(rest)...
		};

		return (*this)[flat_offset(indices)];
	}

	/**
	 * @brief Returns an element using unchecked vector-based indices.
	 * @param indices One normalized index per tensor dimension.
	 * @pre `indices.size() == rank()` and every component is within its
	 *      corresponding dimension.
	 * @complexity O(rank()).
	 */
	reference operator()(const std::vector<size_type>& indices) {return (*this)[flat_offset(indices)];}

	/**
	 * @brief Returns an element using unchecked vector-based indices.
	 * @param indices One normalized index per tensor dimension.
	 * @pre `indices.size() == rank()` and every component is within its
	 *      corresponding dimension.
	 * @complexity O(rank()).
	 */
	const_reference operator()(const std::vector<size_type>& indices) const {return (*this)[flat_offset(indices)];}

	/**
	 * @brief Returns an element using checked variadic indices.
	 *
	 * Each signed component is normalized independently. Negative values count
	 * backward from the end of their corresponding dimensions.
	 *
	 * @tparam Rest Integral types of the remaining index components.
	 * @param first First signed index component.
	 * @param rest Remaining signed index components.
	 * @throws Exceptions::IndexError If the number of components differs from
	 *         `rank()` or any component is out of bounds.
	 * @complexity O(rank()).
	 */
	template<typename... Rest>
	requires ((std::is_integral_v<Rest>) && ...)
	reference at(difference_type first, Rest... rest)
	{
		std::array<difference_type, sizeof...(Rest) + 1> raw_indices{
			first,
			static_cast<difference_type>(rest)...
		};

		return (*this)[normalized_flat_offset(
			raw_indices,
			"Tensor multi-index rank must match tensor rank.",
			"Tensor multi-index component is out of bounds.")];
	}

	/**
	 * @brief Returns an element using checked variadic indices.
	 *
	 * Each signed component is normalized independently. Negative values count
	 * backward from the end of their corresponding dimensions.
	 *
	 * @tparam Rest Integral types of the remaining index components.
	 * @param first First signed index component.
	 * @param rest Remaining signed index components.
	 * @throws Exceptions::IndexError If the number of components differs from
	 *         `rank()` or any component is out of bounds.
	 * @complexity O(rank()).
	 */
	template<typename... Rest>
	requires ((std::is_integral_v<Rest>) && ...)
	const_reference at(difference_type first, Rest... rest) const
	{
		std::array<difference_type, sizeof...(Rest) + 1> raw_indices{
			first,
			static_cast<difference_type>(rest)...
		};

		return (*this)[normalized_flat_offset(
			raw_indices,
			"Tensor multi-index rank must match tensor rank.",
			"Tensor multi-index component is out of bounds.")];
	}

	/**
	 * @brief Returns an element using checked vector-based indices.
	 * @param raw_indices One signed index per tensor dimension.
	 * @throws Exceptions::IndexError If the number of indices differs from
	 *         `rank()` or any component is out of bounds.
	 * @complexity O(rank()).
	 */
	reference at(const std::vector<difference_type>& raw_indices)
	{
		return (*this)[normalized_flat_offset(
			raw_indices,
			"Tensor multi-index rank must match tensor rank.",
			"Tensor multi-index component is out of bounds.")];
	}

	/**
	 * @brief Returns an element using checked vector-based indices.
	 * @param raw_indices One signed index per tensor dimension.
	 * @throws Exceptions::IndexError If the number of indices differs from
	 *         `rank()` or any component is out of bounds.
	 * @complexity O(rank()).
	 */
	const_reference at(const std::vector<difference_type>& raw_indices) const
	{
		return (*this)[normalized_flat_offset(
			raw_indices,
			"Tensor multi-index rank must match tensor rank.",
			"Tensor multi-index component is out of bounds.")];
	}

	/**
	 * @brief Exchanges storage and layout metadata with @p other.
	 * @param other Tensor to exchange with this tensor.
	 * @complexity O(1).
	 */
	void swap(Tensor& other) noexcept {core::ArrayBase<T>::swap(other);}

	/**
	 * @brief Exchanges two tensors using argument-dependent lookup.
	 * @param lhs First tensor.
	 * @param rhs Second tensor.
	 * @complexity O(1).
	 */
	friend void swap(Tensor& lhs, Tensor& rhs) noexcept {lhs.swap(rhs);}
};

} // namespace stratax::container
