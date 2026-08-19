# =============================================================================
# Python
# =============================================================================

find_package(
    Python
    REQUIRED
    COMPONENTS
        Interpreter
        Development.Module
)

if(NOT Python_FOUND)
    message(WARNING "Python interpreter/development module not found; disabling STRATAX_BUILD_PYTHON_BINDINGS.")
    set(STRATAX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "Build the Stratax Python extension" FORCE)
    return()
endif()

execute_process(
    COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
    RESULT_VARIABLE pybind11_cmakedir_result
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(pybind11_cmakedir_result EQUAL 0 AND pybind11_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")
endif()

find_package(pybind11 CONFIG REQUIRED)

if(NOT pybind11_FOUND)
    message(WARNING "pybind11 not found; disabling STRATAX_BUILD_PYTHON_BINDINGS.")
    set(STRATAX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "Build the Stratax Python extension" FORCE)
    return()
endif()

# =============================================================================
# Python Extension
# =============================================================================

pybind11_add_module(
    _core

    bindings/module.cpp
    bindings/shape.cpp
    bindings/boolean_arrays.cpp
    bindings/vector.cpp
    bindings/matrix.cpp
    bindings/tensor.cpp
    bindings/conversions.cpp
    bindings/creation.cpp
    bindings/reductions.cpp
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

install(

    FILES
        ${CMAKE_SOURCE_DIR}/python/stratax/__init__.py
        ${CMAKE_SOURCE_DIR}/python/stratax/_core.pyi
        ${CMAKE_SOURCE_DIR}/python/stratax/py.typed

    DESTINATION
        stratax
)
