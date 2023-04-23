cmake_minimum_required(VERSION 3.1)

set(PROJECT_NAME eigen_premier)
project(${PROJECT_NAME} CXX)

if (NOT CMAKE_BUILD_TYPE) 
  set(CMAKE_BUILD_TYPE Release) 
endif() 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(libs/eigen/install.txt)

add_executable(${PROJECT_NAME} "${PROJECT_SOURCE_DIR}/src/main.cpp")

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic)