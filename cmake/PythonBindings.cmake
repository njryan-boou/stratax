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
    CONFIG
    REQUIRED
)

# =============================================================================
# Python Extension
# =============================================================================

pybind11_add_module(
    _core

    bindings/module.cpp
    bindings/shape.cpp
    bindings/vector.cpp
    bindings/matrix.cpp
    bindings/tensor.cpp
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

        ${CMAKE_SOURCE_DIR}/python/stratax/
        $<TARGET_FILE_NAME:_core>
)

install(

    TARGETS
        _core

    LIBRARY DESTINATION
        stratax

    RUNTIME DESTINATION
        stratax
)