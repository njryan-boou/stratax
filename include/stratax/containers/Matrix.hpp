#pragma once

#include <array>
#include <cstddef>
#include <initializer_list>

#include <stratax/concepts/Numeric.hpp>
#include <stratax/core/ArrayBase.hpp>
#include <stratax/core/Shape.hpp>
#include <stratax/core/validation/Validation.hpp>
#include <stratax/exceptions/Exceptions.hpp>

namespace stratax::container {

template<typename T>
requires Numeric<T>
class Matrix : public core::ArrayBase<T>
{
public:
	using value_type = typename core::ArrayBase<T>::value_type;
	using size_type = typename core::ArrayBase<T>::size_type;
	using difference_type = typename core::ArrayBase<T>::difference_type;
	using reference = typename core::ArrayBase<T>::reference;
	using const_reference = typename core::ArrayBase<T>::const_reference;
	using pointer = typename core::ArrayBase<T>::pointer;
	using const_pointer = typename core::ArrayBase<T>::const_pointer;
	using iterator = typename core::ArrayBase<T>::iterator;
	using const_iterator = typename core::ArrayBase<T>::const_iterator;
	using reverse_iterator = typename core::ArrayBase<T>::reverse_iterator;
	using const_reverse_iterator = typename core::ArrayBase<T>::const_reverse_iterator;

private:
	static core::Shape initializer_shape(
		std::initializer_list<std::initializer_list<value_type>> list)
	{
		const size_type rows = list.size();
		const size_type cols = rows == 0 ? 0 : list.begin()->size();

		for (const auto& row : list)
		{
			if (row.size() != cols)
			{
				throw Exceptions::ShapeError(
					"Matrix initializer rows must all have the same number of columns.");
			}
		}

		return core::Shape{rows, cols};
	}

protected:
	using core::ArrayBase<T>::normalized_flat_offset;

public:
	using core::ArrayBase<T>::at;

	Matrix() : Matrix(0, 0) {}

	Matrix(size_type rows, size_type cols)
		: core::ArrayBase<T>(core::Shape({rows, cols}))
	{}

	Matrix(size_type rows, size_type cols, const_reference value)
		: core::ArrayBase<T>(core::Shape({rows, cols}), value)
	{}

	explicit Matrix(const core::Shape& shape)
		: core::ArrayBase<T>(shape)
	{
		if (shape.rank() != 2)
		{
			throw Exceptions::ShapeError(
				"Matrix shape must be rank 2.");
		}
	}

	Matrix(std::initializer_list<std::initializer_list<value_type>> list)
		: core::ArrayBase<T>(initializer_shape(list))
	{
		size_type index = 0;

		for (const auto& row : list)
		{
			for (const auto& value : row)
			{
				(*this)[index++] = value;
			}
		}
	}

	[[nodiscard]] size_type rows() const noexcept {return this->shape()[0];}
	[[nodiscard]] size_type cols() const noexcept {return this->shape()[1];}

	reference operator()(size_type row, size_type col) {return (*this)[row * cols() + col];}
	const_reference operator()(size_type row, size_type col) const {return (*this)[row * cols() + col];}

	reference at(difference_type row, difference_type col) {return (*this)[normalized_flat_offset(std::array<difference_type, 2>{row, col})];}
	const_reference at(difference_type row, difference_type col) const {return (*this)[normalized_flat_offset(std::array<difference_type, 2>{row, col})];}

	void swap(Matrix& other) noexcept {core::ArrayBase<T>::swap(other);}
};

} // namespace stratax::container
