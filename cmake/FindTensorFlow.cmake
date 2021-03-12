###############################################################################
# Find TensorFlow SDK
#
# Exports the target TensorFlow::Prebuilt
# Recommended method to link is using target_link_libraries(... TensorFlow::Prebuilt).
# No extra steps are required if using this method.
# 
# Alternative, can also use old-style by making use of the following defined variables:
# TensorFlow_FOUND - True if TensorFlow SDK was found.
# TensorFlow_LIBRARIES - Libraries for TensorFlow SDK.
# TensorFlow_INCLUDE_DIRS - Directories containing the TensorFlow SDK include files.

# Check that it's Windows and 64-bit.
# TODO: Implement non-Windows support. Just need to change some default search paths and different library file extensions
if (NOT WIN32)
    message(SEND_ERROR "FindTensorFlow.cmake does not yet implement support other than Windows 64-bit")
endif()

# Expose the ROOT_DIR variable
find_path(
	TensorFlow_ROOT_DIR
	NAMES /include/tensorflow/c/c_api.h
	PATHS "C:/SDK/libtensorflow-gpu-windows-x86_64-1.14.0"
)

# Find include directory
find_path(
    TensorFlow_HEADERS
    NAMES tensorflow/c/c_api.h
    PATHS ${TensorFlow_ROOT_DIR}/include
)

# Find libraries
find_path(
    TensorFlow_LIBDIR
    NAMES tensorflow.dll tensorflow.lib
    PATHS ${TensorFlow_ROOT_DIR}/lib
)
find_library(
    TensorFlow_LIB
    NAMES tensorflow.lib
    PATHS ${TensorFlow_LIBDIR}
)
find_file(TensorFlow_DLL
    NAMES tensorflow.dll
    PATHS ${TensorFlow_LIBDIR}
)

mark_as_advanced(
    TensorFlow_FOUND
    TensorFlow_HEADERS
    TensorFlow_LIBDIR
    TensorFlow_LIB
    TensorFlow_DLL
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorFlow
    DEFAULT_MSG
    TensorFlow_HEADERS
    TensorFlow_DLL
    TensorFlow_LIB
)

if(TensorFlow_FOUND)
    set(TensorFlow_INCLUDE_DIRS ${TensorFlow_HEADERS})
    set(TensorFlow_LIBRARIES ${TensorFlow_LIB})
endif()

if(TensorFlow_FOUND AND NOT TARGET TensorFlow::Prebuilt)
    add_library(TensorFlow::Prebuilt UNKNOWN IMPORTED)
    set_target_properties(TensorFlow::Prebuilt PROPERTIES
        IMPORTED_LOCATION ${TensorFlow_LIB}
        INTERFACE_INCLUDE_DIRECTORIES ${TensorFlow_HEADERS}
    )
    set_target_properties(TensorFlow::Prebuilt PROPERTIES SCSDK_LIBCOPY_SRC ${TensorFlow_DLL})
endif()
