cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# include this to handle the QUIETLY and REQUIRED arguments
include(FindPackageHandleStandardArgs)
include(GetPrerequisites)

#   SIONlib_<api>_<lang>_COMPILE_FLAGS
#   SIONlib_<api>_<lang>_INCLUDE_DIRS
#   SIONlib_<api>_<lang>_LIBRARIES

function (_SIONLib_interrogate_sionconfig api lang)
  string(TOLOWER ${api} SIONCONFIG_API_PARAM)
  string(TOLOWER ${lang} SIONCONFIG_LANG_PARAM)

  execute_process(
    COMMAND "${SIONCONFIG}" "--${SIONCONFIG_API_PARAM}" "--${SIONCONFIG_LANG_PARAM}" "--cflags" "--foobar"
    OUTPUT_VARIABLE SIONCONFIG_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE SIONCONFIG_OUTPUT ERROR_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE SIONCONFIG_RETURN)

  if (NOT "${SIONCONFIG_RETURN}" EQUAL 0)
    if (NOT ${SIONlib_FIND_QUIETLY})
      message(STATUS "Unable to interrogate sionconfig (${SIONCONFIG})")
      message(STATUS ${SIONCONFIG_OUTPUT})
    endif()
    set(SIONCONFIG_OUTPUT)
  endif ()

  if (SIONCONFIG_OUTPUT)
    string(REGEX MATCHALL "(^| )-D([^\" ]+|\"[^\"]+\")" SIONlib_ALL_COMPILE_FLAGS "${SIONCONFIG_OUTPUT}")

    set(SIONlib_COMPILE_FLAGS_WORK)
    foreach (FLAG ${SIONlib_ALL_COMPILE_FLAGS})
      if (SIONlib_COMPILE_FLAGS_WORK)
        set(SIONlib_COMPILE_FLAGS_WORK "${SIONlib_COMPILE_FLAGS_WORK} ${FLAG}")
      else ()
        set(SIONlib_COMPILE_FLAGS_WORK ${FLAG})
      endif ()
    endforeach ()

    string(REGEX MATCHALL "(^| )-I([^\" ]+|\"[^\"]+\")" SIONlib_ALL_INCLUDE_DIRS "${SIONCONFIG_OUTPUT}")
    foreach (IDIR ${SIONlib_ALL_INCLUDE_DIRS})
      string(REGEX REPLACE "^ ?-I" "" IDIR ${IDIR})
      string(REGEX REPLACE "//" "/" IDIR ${IDIR})
      list(APPEND SIONlib_INCLUDE_DIRS_WORK ${IDIR})
    endforeach ()

    set(SIONlib_${api}_${lang}_COMPILE_FLAGS ${SIONlib_COMPILE_FLAGS_WORK} CACHE STRING "SIONlib ${api} API ${lang} compilation flags" FORCE)
    set(SIONlib_${api}_${lang}_INCLUDE_DIRS ${SIONlib_INCLUDE_DIRS_WORK} CACHE STRING "SIONlib ${api} API ${lang} include path" FORCE)
    mark_as_advanced(SIONlib_${api}_${lang}_COMPILE_FLAGS SIONlib_${api}_${lang}_INCLUDE_DIRS)
  endif ()

  execute_process(
    COMMAND "${SIONCONFIG}" "--${SIONCONFIG_API_PARAM}" "--${SIONCONFIG_LANG_PARAM}" "--libs"
    OUTPUT_VARIABLE SIONCONFIG_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE SIONCONFIG_OUTPUT ERROR_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE SIONCONFIG_RETURN)

  if (NOT "${SIONCONFIG_RETURN}" EQUAL 0)
    if (NOT ${SIONlib_FIND_QUIETLY})
      message(STATUS "Unable to interrogate sionconfig (${SIONCONFIG})")
      message(STATUS ${SIONCONFIG_OUTPUT})
    endif()
    set(SIONCONFIG_OUTPUT)
  endif ()

  if (SIONCONFIG_OUTPUT)
    string(REGEX MATCHALL "(^| |-Wl,)-L([^\" ]+|\"[^\"]+\")" SIONlib_ALL_LINK_DIRS "${SIONCONFIG_OUTPUT}")

    set(SIONlib_LINK_DIR)
    foreach (LDIR ${SIONlib_ALL_LINK_DIRS})
      string(REGEX REPLACE "^(| |-Wl,)-L" "" LDIR ${LDIR})
      string(REGEX REPLACE "//" "/" LDIR ${LDIR})
      list(APPEND SIONlib_LINK_DIR ${LDIR})
    endforeach ()

    if (DEFINED CMAKE_${lang}_IMPLICIT_LINK_DIRECTORIES)
      list(APPEND SIONlib_LINK_DIR "${CMAKE_${lang}_IMPLICIT_LINK_DIRECTORIES}")
    endif ()

    string(REGEX MATCHALL "(^| )(-l([^\" ]+|\"[^\"]+\")|[^ ]*\\.(a|so))( |$)" SIONlib_LIBNAMES "${SIONCONFIG_OUTPUT}")

    foreach (LIB ${SIONlib_LIBNAMES})
      string(REGEX REPLACE "^ +" "" LIB ${LIB})
      string(REGEX REPLACE "^-l *" "" LIB ${LIB})
      string(REGEX REPLACE " $" "" LIB ${LIB})

      set(SIONlib_LIB "SIONlib_LIB-NOTFOUND" CACHE FILEPATH "Cleared" FORCE)
      if (EXISTS "${LIB}")
        set(SIONlib_LIB "${LIB}")
      else ()
        find_library(SIONlib_LIB NAMES ${LIB} HINTS ${SIONlib_LINK_DIR})
      endif ()

      if (SIONlib_LIB)
        list(APPEND SIONlib_LIBRARIES_WORK ${SIONlib_LIB})
      endif()
    endforeach ()

    set(SIONlib_${api}_${lang}_LIBRARIES ${SIONlib_LIBRARIES_WORK} CACHE STRING "SIONlib ${api} API ${lang} libraries to link against" FORCE)
    set(SIONlib_LIB "SIONlib_LIB-NOTFOUND" CACHE INTERNAL "Scratch variable for SIONlib lib detection" FORCE)
    mark_as_advanced(SIONlib_${api}_${lang}_LIBRARIES SIONlib_LIB)
  endif ()
endfunction ()


## The function that actually looks for SIONLib

function(_SIONLib_find_sionlib)

  ### Setup the default components
  
  if("x${SIONLib_FIND_COMPONENTS}" STREQUAL "x")
    foreach (api SER OMP MPI OMPI)
      foreach (lang C CXX F90 F77)
        list(APPEND SIONLib_FIND_COMPONENTS "${api}_${lang}")
        set("SIONLib_FIND_REQUIRED_${api}_${lang}" FALSE)
      endforeach()
    endforeach()
  endif()

  
  ### Find SIONCONFIG to find the rest
  
  find_program(SIONCONFIG NAMES sionconfig DOC "SIONlib configuration tool.")
  mark_as_advanced(SIONCONFIG)

  
  ### Find all components
  
  set(_SIONLib_REQUIRED_VARS SIONCONFIG)
  foreach(_SIONLib_ONE_COMPONENT ${SIONLib_FIND_COMPONENTS})
    string(REGEX REPLACE "_.*$" "" api "${_SIONLib_ONE_COMPONENT}")
    string(REGEX REPLACE "^.*_" "" lang "${_SIONLib_ONE_COMPONENT}")
    _SIONLib_interrogate_sionconfig(${api} ${lang})
    if("${SIONLib_FIND_REQUIRED_${_SIONLib_ONE_COMPONENT}}")
      list(APPEND _SIONLib_REQUIRED_VARS "SIONlib_${_SIONLib_ONE_COMPONENT}_LIBRARIES" "SIONlib_${_SIONLib_ONE_COMPONENT}_INCLUDE_DIRS")
    endif()
  endforeach()

  
  ### Verify find results
  
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(SIONLib
    REQUIRED_VARS SIONCONFIG ${_SIONLib_REQUIRED_VARS}
    HANDLE_COMPONENTS
  )

  
  ### Generate the target if we found SIONLib
  
  if("${SIONLib_FOUND}")
    foreach(_SIONLib_ONE_COMPONENT ${SIONLib_FIND_COMPONENTS})
      if("${SIONlib_${api}_${lang}_LIBRARIES}" AND "${SIONlib_${api}_${lang}_INCLUDE_DIRS}" AND NOT TARGET "SIONLib::sion_${api}_${lang}")
        add_library("SIONLib::sion_${api}_${lang}" IMPORTED GLOBAL)
        set_property(TARGET "SIONLib::sion_${api}_${lang}" PROPERTY IMPORTED_LOCATION "${SIONlib_${api}_${lang}_LIBRARIES}")
        target_include_directories("SIONLib::sion_${api}_${lang}" INTERFACE "${SIONlib_${api}_${lang}_INCLUDE_DIRS}")
      endif()
    endforeach()
  endif()
  
endfunction()


## Do the actual lookup

_SIONLib_find_sionlib()
