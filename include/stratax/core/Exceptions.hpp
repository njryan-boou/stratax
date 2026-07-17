#pragma once

#include <stdexcept>

namespace Exceptions {

	/** @brief Base class for Stratax-specific runtime errors. */
	class StrataxError : public std::runtime_error
	{
	public:
		using std::runtime_error::runtime_error;
	};

	/** @brief Signals an invalid or incompatible shape. */
	class ShapeError : public StrataxError
	{
	public:
		using StrataxError::StrataxError;
	};

	/** @brief Signals an invalid dimension count or dimension arithmetic failure. */
	class DimensionError : public StrataxError
	{
	public:
		using StrataxError::StrataxError;
	};

	/** @brief Signals an invalid index access. */
	class IndexError : public StrataxError
	{
	public:
		using StrataxError::StrataxError;
	};

	/** @brief Signals an invalid or unsupported type. */
	class TypeError : public StrataxError
	{
	public:
		using StrataxError::StrataxError;
	};

	/** @brief Signals that an operation cannot be completed with the current broadcast rules. */
	class BroadcastError : public StrataxError
	{
	public:
		using StrataxError::StrataxError;
	};

	/** @brief Signals a division by zero. */
	class ZeroDivisionError : public StrataxError
	{
	public:
		using StrataxError::StrataxError;
	};
}
