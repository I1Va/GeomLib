cmake_minimum_required(VERSION 3.10)
project(GeomLibTests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------
# Library source files
# -----------------------------
set(LIB_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/Geom.cpp   # all library .cpp files
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/IntinsicGeom.cpp
)

# -----------------------------
# Test sources
# -----------------------------
set(TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/IntinsicGeomUnit.cpp
)

# -----------------------------
# Create test executable
# -----------------------------
add_executable(${PROJECT_NAME} ${TEST_SOURCES} ${LIB_SOURCES})

# Include directories for library headers
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

# Compile flags
target_compile_options(${PROJECT_NAME} PRIVATE -march=native -mtune=native)

# -----------------------------
# Google Test
# -----------------------------
find_package(GTest REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE gtest gtest_main)

# Address sanitizer (optional)
target_compile_options(${PROJECT_NAME} PRIVATE -fsanitize=address -fno-omit-frame-pointer)
target_link_options(${PROJECT_NAME} PRIVATE -fsanitize=address)

# -----------------------------
# Enable testing
# -----------------------------
enable_testing()
include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME})
