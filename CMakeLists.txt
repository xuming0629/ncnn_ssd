cmake_minimum_required(VERSION 2.8.12)
project(ncnn_ssd CXX C)

message(STATUS " this is BINARY dir " ${PROJECT_BINARY_DIR})
message(STATUS " this is SOURCE_dir " ${PROJECT_SOURCE_DIR})

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)


add_subdirectory(ncnn_ssd)

