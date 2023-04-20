cmake_minimum_required(VERSION 3.1)

set(PROJECT_NAME functional_programming)
project(${PROJECT_NAME} CXX)

if (NOT CMAKE_BUILD_TYPE) 
  set(CMAKE_BUILD_TYPE Release) 
endif() 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable("using_algorithms" "${PROJECT_SOURCE_DIR}/src/using_algorithms.cpp")
target_compile_options("using_algorithms" PRIVATE -Wall -Wextra -pedantic)

add_executable("using_functional" "${PROJECT_SOURCE_DIR}/src/using_functional.cpp")
target_compile_options("using_algorithms" PRIVATE -Wall -Wextra -pedantic)

add_executable("using_numeric" "${PROJECT_SOURCE_DIR}/src/using_numeric.cpp")
target_compile_options("using_algorithms" PRIVATE -Wall -Wextra -pedantic)