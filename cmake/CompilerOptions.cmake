# =============================================================================
# Compiler Options
# =============================================================================

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable /FS flag for MSVC to allow parallel compilation with shared PDB
if(MSVC)
    string(APPEND CMAKE_CXX_FLAGS " /FS")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR}/bin
)