cmake_minimum_required(VERSION 2.8)
project( task_3 )
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")

#FILE(GLOB_RECURSE LibFiles "include/*.h")
#add_custom_target(headers SOURCES ${LibFiles})

FIND_PACKAGE(Qt4 REQUIRED)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# add_executable( training.out main.cpp )
aux_source_directory(. DIR_SRCS)
add_executable( task_3.out ${DIR_SRCS})

target_link_libraries( task_3.out ${OpenCV_LIBS} )
