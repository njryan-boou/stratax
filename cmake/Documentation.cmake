# =============================================================================
# Documentation
# =============================================================================

find_package(Doxygen QUIET)

if(DOXYGEN_FOUND)

    add_custom_target(
        docs

        COMMAND
            ${DOXYGEN_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/Doxyfile

        WORKING_DIRECTORY
            ${CMAKE_SOURCE_DIR}

        COMMENT
            "Generating Stratax documentation"

        VERBATIM
    )

endif()