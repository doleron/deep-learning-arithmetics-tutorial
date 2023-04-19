cmake_minimum_required(VERSION 3.1)

set(PROJECT_NAME eigen_premier)
project(${PROJECT_NAME} CXX)

# set the default build type to release
if (NOT CMAKE_BUILD_TYPE) 
  set(CMAKE_BUILD_TYPE Release) 
endif() 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# downloading eigen
include(libs/eigen/install.txt)

include_directories(include)

link_directories(${CMAKE_PREFIX_PATH}/lib/)

file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")
add_executable(${PROJECT_NAME} ${SOURCES})

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic)