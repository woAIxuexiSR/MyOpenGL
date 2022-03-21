cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 52;70;75)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)


if(NOT DEFINED CMAEK_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

project(MyOpenGL LANGUAGES CXX CUDA)

message("PROJECT_NAME : " ${PROJECT_NAME})


if(NOT DEFINED HELPER_CUDA_DIR)
    set(HELPER_CUDA_DIR ~/cuda-samples/Common/ CACHE PATH "helper_cuda path")
endif()
if(NOT DEFINED STB_DIR)
    set(STB_DIR ~/stb/ CACHE PATH "stb path")
endif()

include_directories(${HELPER_CUDA_DIR})
include_directories(${STB_DIR})


add_subdirectory(lib)

aux_source_directory(src sources)
add_executable(main ${sources})

target_include_directories(main PUBLIC "include")

target_compile_options(main PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)