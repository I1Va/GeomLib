find_package(OpenMP REQUIRED)

add_library(GeomLib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Geom.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/IVec3d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/IVec3f.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/GmUtilities.cpp
)

if(GEOMLIB_BUILD_TESTS)
    add_subdirectory(tests)
endif()

target_compile_options(GeomLib PUBLIC -march=native -mtune=native)

target_include_directories(GeomLib
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

target_link_libraries(GeomLib 
    PRIVATE OpenMP::OpenMP_CXX
)

