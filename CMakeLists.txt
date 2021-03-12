#
#
# Rafael Radkowski
# rafael@iastate.edu
# May 16, 2016

cmake_minimum_required(VERSION 3.12)
cmake_policy(SET CMP0074 NEW)

project(graph_loader)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

################################################################

add_subdirectory(src_1)