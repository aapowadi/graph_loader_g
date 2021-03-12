###############################################################################
# Find Structure Core SDK
#
# Exports the target Structure::Prebuilt
# Recommended method to link is using target_link_libraries(... Structure::Prebuilt).
# No extra steps are required if using this method.
# 
# Alternative, can also use old-style by making use of the following defined variables:
# StructureSDK_FOUND - True if Structure SDK was found.
# StructureSDK_LIBRARIES - Libraries for Structure SDK.
# StructureSDK_INCLUDE_DIRS - Directories containing the Structure SDK include files.

# Check that it's Windows and 64-bit.
# TODO: Implement non-Windows support. Just need to change some default search paths and different library file extensions
if (NOT WIN32)
    message(SEND_ERROR "FindStructureSDK.cmake does not yet implement support other than Windows 64-bit")
endif()

# Expose the ROOT_DIR variable
find_path(
	StructureSDK_ROOT_DIR
	NAMES /Libraries/Structure/Headers/ST/CaptureSession.h
	PATHS "C:/SDK/StructureSDK-CrossPlatform-0.7.2"
)

# Find include directory
find_path(
    StructureSDK_HEADERS
    NAMES ST/CaptureSession.h
    PATHS ${StructureSDK_ROOT_DIR}/Libraries/Structure/Headers
)

# Find libraries
find_path(
    StructureSDK_LIBDIR
    NAMES Structure.dll Structure.lib
    PATHS ${StructureSDK_ROOT_DIR}/Libraries/Structure/Windows/x86_64
)
find_library(
    StructureSDK_LIB
    NAMES Structure.lib
    PATHS ${StructureSDK_LIBDIR}
)
find_file(StructureSDK_DLL
    NAMES Structure.dll
    PATHS ${StructureSDK_LIBDIR}
)

mark_as_advanced(
    StructureSDK_FOUND
    StructureSDK_HEADERS
    StructureSDK_LIBDIR
    StructureSDK_LIB
    StructureSDK_DLL
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(StructureSDK
    DEFAULT_MSG
    StructureSDK_HEADERS
    StructureSDK_DLL
    StructureSDK_LIB
)

if(StructureSDK_FOUND)
    set(StructureSDK_INCLUDE_DIRS ${StructureSDK_HEADERS})
    set(StructureSDK_LIBRARIES ${StructureSDK_LIB})
endif()

if(StructureSDK_FOUND AND NOT TARGET Structure::Prebuilt)
    add_library(Structure::Prebuilt UNKNOWN IMPORTED)
    set_target_properties(Structure::Prebuilt PROPERTIES
        IMPORTED_LOCATION ${StructureSDK_LIB}
        INTERFACE_INCLUDE_DIRECTORIES ${StructureSDK_HEADERS}
    )
    set_target_properties(Structure::Prebuilt PROPERTIES SCSDK_LIBCOPY_SRC ${StructureSDK_DLL})
endif()
