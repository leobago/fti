# Author: Kai Keller, 2017
# Package to link the Lustre API

# Sets variables:
#    LUSTREAPI_FOUND
#    LUSTREAPI_INCLUDE_DIRS
#    LUSTREAPI_LIBRARIES
#    LUSTREAPI_DEFINITIONS

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)


## The function that actually looks for LUSTRE

function(_LUSTREAPI_Find_lustre)
    
    ### Find the header path
    
    find_path(LUSTREAPI_INCLUDE_DIR liblustreapi.h 
        PATHS /usr/include/lustre
        HINTS ${LUSTREAPI_CMAKE_INCLUDE_DIRS}
    )
    mark_as_advanced(LUSTREAPI_INCLUDE_DIR)
    
    
    ### Find the library
    
    find_library(LUSTREAPI_LIBRARY
        NAMES liblustreapi.a ${LUSTREAPI_CMAKE_LIBRARIES}
        HINTS ${LUSTREAPI_CMAKE_LIBRARY_DIRS}
    )
    mark_as_advanced(LUSTREAPI_LIBRARY)
    
    
    ### Verify find results

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(LUSTREAPI
        REQUIRED_VARS LUSTREAPI_INCLUDE_DIR LUSTREAPI_LIBRARY
    )


    ## Generate the target if we found Lustre
    
    if("${LUSTREAPI_FOUND}")
        if(NOT TARGET LUSTREAPI::lustre)
            add_library(LUSTREAPI::lustre IMPORTED GLOBAL)
            set_property(TARGET LUSTREAPI::lustre PROPERTY IMPORTED_LOCATION "${LUSTREAPI_LIBRARY}")
            target_include_directories(LUSTREAPI::lustre INTERFACE "${LUSTREAPI_INCLUDE_DIR}")
        endif()
    else()
        unset(LUSTREAPI_INCLUDE_DIR CACHE)
        unset(LUSTREAPI_LIBRARY CACHE)
    endif()

endfunction()


## Do the actual lookup

_LUSTREAPI_Find_lustre()
