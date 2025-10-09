cmake_minimum_required(VERSION 3.10)

project(GeomLib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(GeomLib STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/Geom.cpp
    main.cpp
)

target_include_directories(GeomLib
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inc
)