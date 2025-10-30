enable_testing()

find_package(GTest REQUIRED)
include(GoogleTest)

set(LIB_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/Geom.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/IVec3d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/IVec3f.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/GmUtilities.cpp
)
set(TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_IVec3d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_IVec3f.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_GmUtilities.cpp
)

add_executable(GeomLibUnitTests ${TEST_SOURCES} ${LIB_SOURCES})

target_include_directories(GeomLibUnitTests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

target_compile_options(GeomLibUnitTests PRIVATE -march=native -mtune=native)
target_link_libraries(GeomLibUnitTests PRIVATE gtest gtest_main)

target_compile_options(GeomLibUnitTests PRIVATE -fsanitize=address -fno-omit-frame-pointer)
target_link_options(GeomLibUnitTests PRIVATE -fsanitize=address)

gtest_discover_tests(GeomLibUnitTests)