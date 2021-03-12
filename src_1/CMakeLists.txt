#
#
# Rafael Radkowski
# rafael@iastate.edu
# May 16, 2016


cmake_minimum_required(VERSION 3.12)
cmake_policy(SET CMP0074 NEW)

project(graph_loader)


# Set the project architecture for this example
set(PROJECT_ARCH "x64")

# find_path(
    # TF_C_API_DIR
    # NAMES "lib/tensorflow.lib" "include/tensorflow/c/c_api.h"
# )

find_package(StructureSDK REQUIRED)
find_package(TensorFlow REQUIRED)
find_package(OpenCV REQUIRED)
# All output files are copied to bin
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)


################################################################
# sources

set(MAIN_SRC
	main.cpp

)

set(Camera_SRC
	StructureCoreCaptureDevice.cpp
	StructureCoreCaptureDevice.h
	
)

set(PoseEst_SRC
	Pose_Network.cpp
	Pose_Network.h
	tf_utils.hpp
	tf_utils.cpp
)  

################################################################
#  SRC Groups
source_group(Main FILES ${MAIN_SRC})
source_group(Camera FILES ${Camera_SRC})
source_group(PoseEst FILES ${PoseEst_SRC})

#link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
link_directories(${OPENCV_LIBRARY_DIR})


SET("CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG" "../bin")
SET("CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE" "../bin")

################################################################

add_executable(graph_loader
	${MAIN_SRC}
	${Camera_SRC}
	${PoseEst_SRC}
)

# add a "d" to all debug libraries
SET_TARGET_PROPERTIES(graph_loader PROPERTIES  DEBUG_POSTFIX "d")
SET_TARGET_PROPERTIES(graph_loader PROPERTIES LINK_FLAGS_RELEASE " /FORCE:MULTIPLE")
SET_TARGET_PROPERTIES(graph_loader PROPERTIES LINK_FLAGS_DEBUG "/FORCE:MULTIPLE ")
SET_TARGET_PROPERTIES(graph_loader PROPERTIES LINK_FLAGS "/FORCE:MULTIPLE")

target_link_libraries(graph_loader
	# Structure SDK
	Structure::Prebuilt
	#Tensorflow
	TensorFlow::Prebuilt
	#OpenCV
	${OpenCV_LIBS}
)

################################################################
