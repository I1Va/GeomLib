cmake_minimum_required(VERSION 3.10)

project(GeomLib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(GeomLib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Geom.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/IVec3d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/IVec3f.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/GmUtilities.cpp
)

option(GEOMLIB_BUILD_TESTS "Build GeomLib unit tests" ON)
if(GEOMLIB_BUILD_TESTS)
    add_subdirectory(tests)
endif()


target_compile_options(GeomLib PUBLIC -march=native -mtune=native)

target_include_directories(GeomLib
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inc
)