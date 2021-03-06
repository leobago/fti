# Author: Kai Keller, 2017
# Package to link the Lustre API

# Sets variables:
#    LUSTREAPI_FOUND
#    LUSTREAPI_INCLUDE_DIRS
#    LUSTREAPI_LIBRARIES
#    LUSTREAPI_DEFINITIONS
set(
    LUSTREAPI_DEFINITIONS -DLUSTRE ${LUSTREAPI_CMAKE_DEFINITIONS})
find_path(
    LUSTREAPI_INCLUDE_DIR liblustreapi.h
    HINTS /usr/include/lustre ${LUSTREAPI_CMAKE_INCLUDE_DIRS})
find_library(
    LUSTREAPI_LIBRARY
    NAMES liblustreapi.a ${LUSTREAPI_CMAKE_LIBRARIES}
    HINTS /usr/lib ${LUSTREAPI_CMAKE_LIBRARY_DIRS})
include(
    FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    LUSTREAPI DEFAULT_MSG
    LUSTREAPI_LIBRARY
    LUSTREAPI_INCLUDE_DIR)
mark_as_advanced(
    LUSTREAPI_INCLUDE_DIR
    LUSTREAPI_LIBRARY)
set(
    LUSTREAPI_INCLUDE_DIRS ${LUSTREAPI_INCLUDE_DIR})
set(
    LUSTREAPI_LIBRARIES ${LUSTREAPI_LIBRARY})
