# =============================================================================
# Python
# =============================================================================

find_package(
    Python
    REQUIRED
    COMPONENTS
        Interpreter
        Development
)

find_package(
    pybind11
    REQUIRED
)

# =============================================================================
# Python Extension
# =============================================================================

pybind11_add_module(
    _core

    bindings/module.cpp
    bindings/common/exceptions.cpp
    bindings/common/helpers.cpp
    bindings/common/utilities.cpp
    bindings/shape/bind.cpp
    bindings/shape/constructors.cpp
    bindings/shape/properties.cpp

    bindings/vector/constructors.cpp
    bindings/vector/bind.cpp
    bindings/vector/indexing.cpp
    bindings/vector/arithmetic.cpp
    bindings/vector/conversions.cpp
    bindings/vector/properties.cpp

    bindings/matrix/constructors.cpp
    bindings/matrix/bind.cpp
    bindings/matrix/indexing.cpp
    bindings/matrix/arithmetic.cpp
    bindings/matrix/slicing.cpp
    bindings/matrix/conversions.cpp
    bindings/matrix/properties.cpp

    bindings/tensor/constructors.cpp
    bindings/tensor/bind.cpp
    bindings/tensor/indexing.cpp
    bindings/tensor/slicing.cpp
    bindings/tensor/arithmetic.cpp
    bindings/tensor/conversions.cpp
    bindings/tensor/properties.cpp

    bindings/conversions.cpp
    bindings/creation.cpp
)

target_include_directories(
    _core

    PRIVATE
        include
)

add_custom_command(

    TARGET _core

    POST_BUILD

    COMMAND
        ${CMAKE_COMMAND}
        -E
        copy_if_different

        $<TARGET_FILE:_core>

        ${CMAKE_SOURCE_DIR}/python/stratax/$<TARGET_FILE_NAME:_core>
)

# =============================================================================
# Installation
# =============================================================================

install(

    TARGETS
        _core

    LIBRARY DESTINATION
        stratax

    RUNTIME DESTINATION
        stratax
)