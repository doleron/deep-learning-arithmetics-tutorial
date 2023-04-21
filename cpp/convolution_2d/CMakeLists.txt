cmake_minimum_required(VERSION 3.1)

set(PROJECT_NAME convolution_2d_example)
project(${PROJECT_NAME} CXX)

if (NOT CMAKE_BUILD_TYPE) 
  set(CMAKE_BUILD_TYPE Release) 
endif() 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(libs/eigen/install.txt)

link_directories(${CMAKE_PREFIX_PATH}/lib/)

add_executable(${PROJECT_NAME} "${PROJECT_SOURCE_DIR}/src/convolution_2d_example.cpp")
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic)

add_executable("same_padding_example" "${PROJECT_SOURCE_DIR}/src/same_padding_example.cpp")
target_compile_options("same_padding_example" PRIVATE -Wall -Wextra -pedantic)