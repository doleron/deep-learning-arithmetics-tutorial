cmake_minimum_required(VERSION 3.1)

set(PROJECT_NAME using_tensors)
project(${PROJECT_NAME} CXX)

if (NOT CMAKE_BUILD_TYPE) 
  set(CMAKE_BUILD_TYPE Release) 
endif() 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(libs/eigen/install.txt)

add_executable(${PROJECT_NAME} "${PROJECT_SOURCE_DIR}/src/using_tensors.cpp")
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic -Wno-enum-compare)

add_executable("softmax_tensor" "${PROJECT_SOURCE_DIR}/src/softmax_tensor.cpp")
target_compile_options("softmax_tensor" PRIVATE -Wall -Wextra -pedantic -Wno-enum-compare -Wno-maybe-uninitialized -Wno-uninitialized)
