cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

if(NOT DEFINED CMAEK_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

project(MyOpenGL LANGUAGES CXX)

message("PROJECT_NAME : " ${PROJECT_NAME})

# aux_source_directory(. sources)
add_library(test OBJECT test.cpp)       # STATIC SHARED OBJECT

add_executable(main main.cpp)
target_link_libraries(main PUBLIC test)