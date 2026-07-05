#pragma once

#include <stdexcept>

namespace Exceptions {

    class StrataxError : public std::runtime_error
    {
    public:
        using std::runtime_error::runtime_error;
    };

    class ShapeError : public StrataxError
    {
    public:
        using StrataxError::StrataxError;
    };

    class DimensionError : public StrataxError
    {
    public:
        using StrataxError::StrataxError;
    };

    class IndexError : public StrataxError
    {
    public:
        using StrataxError::StrataxError;
    };

    class DTypeError : public StrataxError
    {
    public:
        using StrataxError::StrataxError;
    };

    class BroadcastError : public StrataxError
    {
    public:
        using StrataxError::StrataxError;
    };

    class ZeroDivisionError : public StrataxError
    {
    public:
        using StrataxError::StrataxError;
    };
}

