cmake_minimum_required(VERSION 3.9)
project(TDCV_ex2)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "version: ${OpenCV_VERSION}")
message(STATUS "libraries: ${OpenCV_LIBS}")
message(STATUS "include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(include ${OpenCV_INCLUDE_DIRS})
include_directories(include ${/home/highstars1/CLionProjects/TDCV_ex2})
set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILES main.cpp)
add_executable(TDCV_ex2 main.cpp task_2.cpp)
target_link_libraries(TDCV_ex2 ${OpenCV_LIBS})




