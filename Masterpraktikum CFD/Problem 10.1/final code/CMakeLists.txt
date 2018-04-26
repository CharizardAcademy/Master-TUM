cmake_minimum_required(VERSION 3.10)
project(CFD_ex1 C)

set(CMAKE_C_STANDARD 99)

add_executable(CFD_ex1 main.c helper.c init.c sor.c visual.c inple.c uvp.c boundary_val.c)