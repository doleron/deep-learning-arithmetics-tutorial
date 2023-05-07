cmake_minimum_required(VERSION 3.1)

set(PROJECT_NAME playing_with_activation_functions)
project(${PROJECT_NAME} CXX)

if (NOT CMAKE_BUILD_TYPE) 
  set(CMAKE_BUILD_TYPE Release) 
endif() 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

include(libs/eigen/install.txt)

add_executable("activation_examples" "${PROJECT_SOURCE_DIR}/src/activation_examples.cpp")
target_compile_options("activation_examples" PRIVATE -Wall -Wextra -pedantic)
