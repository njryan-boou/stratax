#pragma once

#include <ostream>
#include <vector>

#include <stratax/concepts/Numeric.hpp>
#include <stratax/core/Buffer.hpp>
#include <stratax/exceptions/Exceptions.hpp>
#include <stratax/core/validation/Validation.hpp>

namespace stratax::core {

/** @brief Stores dimension lengths for an array shape. */
class Shape
{
private:
	Buffer<std::size_t> dims_;

	void validate_dimensions() const
	{
		// Dimensions are stored as std::size_t, so negative values are not representable.
	}

public:
	/** @brief Tag type documenting that zero-valued dimensions are intentional. */
	struct allow_zero_t {};

	/** @brief Tag value documenting that zero-valued dimensions are intentional. */
	static constexpr allow_zero_t allow_zero{};

	/** @brief Mutable iterator over dimension lengths. */
	using iterator = Buffer<std::size_t>::iterator;

	/** @brief Const iterator over dimension lengths. */
	using const_iterator = Buffer<std::size_t>::const_iterator;

	/** @brief Mutable reverse iterator over dimension lengths. */
	using reverse_iterator = Buffer<std::size_t>::reverse_iterator;

	/** @brief Const reverse iterator over dimension lengths. */
	using const_reverse_iterator = Buffer<std::size_t>::const_reverse_iterator;

	/** @brief Creates an empty shape. */
	Shape() noexcept = default;

	/** @brief Creates a shape from integral dimension lengths. */
	template<Integral... Dims>
	requires (sizeof...(Dims) > 0)
	Shape(Dims... dims)
		: dims_{static_cast<std::size_t>(dims)...}
	{
		validate_dimensions();
	}

	/** @brief Creates a shape from an initializer list of dimensions. */
	Shape(std::initializer_list<std::size_t> dims)
		: dims_(dims)
	{
		validate_dimensions();
	}

	/** @brief Creates a shape from an initializer list with an explicit zero-allowed tag. */
	Shape(std::initializer_list<std::size_t> list, allow_zero_t allow_zero) : dims_(list)
	{
		(void)allow_zero;
	}

	/** @brief Creates a shape by copying dimensions from a buffer. */
	Shape(const Buffer<std::size_t>& dims)
	: dims_(dims)
	{
		validate_dimensions();
	}

	/** @brief Creates a shape from a buffer with an explicit zero-allowed tag. */
	Shape(const Buffer<std::size_t>& dims, allow_zero_t allow_zero)
	: dims_(dims)
	{
		(void)allow_zero;
	}

	/** @brief Creates a shape by moving dimensions from a buffer. */
	Shape(Buffer<std::size_t>&& dims)
	: dims_(std::move(dims))
	{
		validate_dimensions();
	}

	/** @brief Creates a shape by copying dimensions from a std::vector. */
	Shape(const std::vector<std::size_t>& dims)
		: dims_(dims.size())
	{
		for (std::size_t i = 0; i < dims.size(); ++i)
		{
			dims_[i] = dims[i];
		}

		validate_dimensions();
	}

	/** @brief Destroys the shape. */
	~Shape() = default;

	/** @brief Returns the total number of elements implied by the shape. */
	[[nodiscard]]
	std::size_t elements() const
	{
		if (empty())
		{
			return 0;
		}
		std::size_t prod = 1;
		for (std::size_t dim : dims_)
		{
			prod = validation::checked_multiply(prod, dim, "Shape elements overflow");
		}
		return prod;
	}

	/** @brief Returns the number of stored dimensions. */
	[[nodiscard]] std::size_t rank() const
	{
		return dims_.size();
	}

	/** @brief Returns the dimension length at a zero-based index. */
	const std::size_t& operator()(std::size_t index) const
	{
		validation::require_index(index, rank(), "Shape dimension index out of bounds");
		return dims_[index];
	}

	/** @brief Returns the dimension length using signed indexing. */
	const std::size_t& operator[](std::ptrdiff_t index) const
	{
		return dims_[validation::normalize_index(index, rank(), "Shape dimension index out of bounds")];
	}

	/** @brief Returns whether the shape has no dimensions. */
	[[nodiscard]] bool empty() const noexcept
	{
		return dims_.empty();
	}

	/** @brief Compares two shapes for rank and dimension equality. */
	[[nodiscard]] bool operator==(const Shape& other) const noexcept
	{
		if (rank() != other.rank())
		{
			return false;
		}
		for (std::size_t i = 0; i < rank(); ++i)
		{
			if (dims_[i] != other.dims_[i])
			{
				return false;
			}
		}
		return true;
	}

	/** @brief Returns whether two shapes differ in rank or dimension values. */
	[[nodiscard]] bool operator!=(const Shape& other) const noexcept
	{
		return !(*this == other);
	}

	/** @brief Returns an iterator to the first stored dimension. */
	[[nodiscard]] iterator begin() noexcept
	{
		return dims_.begin();
	}

	/** @brief Returns an iterator one past the last stored dimension. */
	[[nodiscard]] iterator end() noexcept
	{
		return dims_.end();
	}

	/** @brief Returns a const iterator to the first stored dimension. */
	[[nodiscard]] const_iterator begin() const noexcept
	{
		return dims_.begin();
	}

	/** @brief Returns a const iterator one past the last stored dimension. */
	[[nodiscard]] const_iterator end() const noexcept
	{
		return dims_.end();
	}

	/** @brief Returns a const iterator to the first stored dimension. */
	[[nodiscard]] const_iterator cbegin() const noexcept
	{
		return dims_.cbegin();
	}

	/** @brief Returns a const iterator one past the last stored dimension. */
	[[nodiscard]] const_iterator cend() const noexcept
	{
		return dims_.cend();
	}

	/** @brief Returns a reverse iterator to the last stored dimension. */
	[[nodiscard]] reverse_iterator rbegin() noexcept
	{
		return dims_.rbegin();
	}

	/** @brief Returns a const reverse iterator to the last stored dimension. */
	[[nodiscard]] const_reverse_iterator rbegin() const noexcept
	{
		return dims_.rbegin();
	}

	/** @brief Returns a const reverse iterator to the last stored dimension. */
	[[nodiscard]] const_reverse_iterator crbegin() const noexcept
	{
		return dims_.crbegin();
	}

	/** @brief Returns a reverse iterator before the first stored dimension. */
	[[nodiscard]] reverse_iterator rend() noexcept
	{
		return dims_.rend();
	}

	/** @brief Returns a const reverse iterator before the first stored dimension. */
	[[nodiscard]] const_reverse_iterator rend() const noexcept
	{
		return dims_.rend();
	}

	/** @brief Returns a const reverse iterator before the first stored dimension. */
	[[nodiscard]] const_reverse_iterator crend() const noexcept
	{
		return dims_.crend();
	}

	/** @brief Swaps the stored dimensions with another shape. */
	void swap(Shape& other) noexcept
	{
		dims_.swap(other.dims_);
	}

};

/** @brief Writes a shape to a stream in tuple-like form. */
inline std::ostream& operator<<(std::ostream& os, const Shape& shape)
{
	os << "(";

	bool first = true;
	for (std::size_t dim : shape)
	{
		if (!first)
			os << ", ";

		os << dim;
		first = false;
	}

	if (shape.rank() == 1)
	{
		os << ",";
	}

	os << ")";

	return os;
}

}
