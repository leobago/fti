cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

if(POLICY CMP0054)
	cmake_policy(SET CMP0054 NEW)
endif()

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
	project(FTI C Fortran)
endif()

## [START] DETERMINE COMPILER MAJOR VERSIONS
string(REGEX MATCH "[0-9]+" CMAKE_C_COMPILER_VERSION_MAJOR ${CMAKE_C_COMPILER_VERSION} )
set(FORTRAN_VER_UNKNOWN 0)
set(ADD_CFLAGS "")

if (NOT CMAKE_Fortran_COMPILER_VERSION)

  message(AUTHOR_WARNING "
  ** Cmake variable 'CMAKE_Fortran_COMPILER_VERSION' is unset
  *  attempt to determine it manually...")

  string(COMPARE EQUAL ${CMAKE_Fortran_COMPILER_ID} "Intel" COMPILER_IDENT_INTEL)
  string(COMPARE EQUAL ${CMAKE_Fortran_COMPILER_ID} "GNU" COMPILER_IDENT_GNU)
  string(COMPARE EQUAL ${CMAKE_Fortran_COMPILER_ID} "PGI" COMPILER_IDENT_PGI)
  if(${COMPILER_IDENT_INTEL})
    set(VER_CHECK_SRC "${CMAKE_SOURCE_DIR}/test/cmake/intel_major_ver.f90")
  elseif(${COMPILER_IDENT_GNU})
    set(VER_CHECK_SRC "${CMAKE_SOURCE_DIR}/test/cmake/gnu_major_ver.f90")
  elseif(${COMPILER_IDENT_PGI})
    set(VER_CHECK_SRC "${CMAKE_SOURCE_DIR}/test/cmake/pgi_major_ver.f90")
  endif()

  set(CMAKE_Fortran_FLAGS "-cpp")
  try_run( PROG_RAN COMPILE_SUCCESS
      "${CMAKE_BINARY_DIR}" "${VER_CHECK_SRC}"
    RUN_OUTPUT_VARIABLE VER_STRING
  )
  if ( COMPILE_SUCCESS )
    string( REGEX MATCH "[0-9]+"
      DETECTED_VER "${VER_STRING}"
    )

  set( CMAKE_Fortran_COMPILER_VERSION_MAJOR "${DETECTED_VER}" )

  message(AUTHOR_WARNING "
  ** The major version was determined as: ${VER_STRING}")
  else()

  set(FORTRAN_VER_UNKNOWN 1)
  set(CMAKE_Fortran_COMPILER_VERSION_MAJOR "")
  message(AUTHOR_WARNING "
  ** The Fortran version could not be determined!")

  endif()
else()

  string(REGEX MATCH "[0-9]+" CMAKE_Fortran_COMPILER_VERSION_MAJOR ${CMAKE_Fortran_COMPILER_VERSION} )

endif()

if(NOT ${FORTRAN_VER_UNKNOWN})

  set(C_COMPILER_IDENT "${CMAKE_C_COMPILER_ID}_${CMAKE_C_COMPILER_VERSION_MAJOR}")
  set(Fortran_COMPILER_IDENT "${CMAKE_Fortran_COMPILER_ID}_${CMAKE_Fortran_COMPILER_VERSION_MAJOR}")
  string(COMPARE NOTEQUAL ${C_COMPILER_IDENT} ${Fortran_COMPILER_IDENT} COMPILER_VER_DIFFER)
  if(${COMPILER_VER_DIFFER})
      message(WARNING "
  ** You are using different compiler idetifications for Fortran and C!
  *  This might lead to undefined behavior!")
  endif()

endif()
## [END] DETERMINE COMPILER MAJOR VERSIONS

option(ENABLE_FORTRAN "Enables the generation of the Fortran wrapper for FTI" ON)
option(ENABLE_EXAMPLES "Enables the generation of examples" ON)
option(ENABLE_SIONLIB "Enables the parallel I/O SIONlib for FTI" OFF)
option(ENABLE_HDF5 "Enables the HDF5 checkpoints for FTI" OFF)
option(ENABLE_TESTS "Enables the generation of tests" ON)
option(ENABLE_LUSTRE "Enables Lustre Support" OFF)
option(ENABLE_DOCU "Enables the generation of a Doxygen documentation" OFF)

set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeScripts")
include(AppendProperty)
include(FortranCInterface)
include(CheckFortranCompilerFlag)
include(CheckCCompilerFlag)

#check for ZLIB (crc32)
find_package(ZLIB)
if(NOT ZLIB_FOUND)
    set(ADD_CFLAGS "${ADD_CFLAGS} -DFTI_NOZLIB")
endif()

find_package(MPI REQUIRED)
if(NOT DEFINED NO_OPENSSL)
	find_package(OPENSSL REQUIRED)
else()
	set(OPENSSL_FOUND false)
endif()

if(ENABLE_LUSTRE)
    find_package(LUSTREAPI)
endif()

add_subdirectory(deps)

include_directories("${CMAKE_CURRENT_BINARY_DIR}"
	"${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${MPI_Fortran_INCLUDE_PATH} ${MPI_C_INCLUDE_PATH})

set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include)

set(SRC_FTI
	src/api.c src/checkpoint.c src/conf.c src/meta.c
	src/postckpt.c src/postreco.c src/recover.c
	src/tools.c src/topo.c src/ftiff.c src/hdf5.c
	src/diff-checkpoint.c src/stage.c src/incremental-checkpoint.c)

# add compiler flags
if("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
  set(ADD_CFLAGS "${ADD_CFLAGS} -Wall")
elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
  set(ADD_CFLAGS "${ADD_CFLAGS} -Wall")
  try_compile(C_COMPILER_HAS_FLAG ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/test/cmake/minimal.c COMPILE_DEFINITIONS "-Wno-format-truncation")
  if(C_COMPILER_HAS_FLAG)
      set(ADD_CFLAGS "${ADD_CFLAGS} -Wno-format-truncation")
  endif()
elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "PGI")
  try_compile(C_COMPILER_HAS_FLAG ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/test/cmake/minimal.c COMPILE_DEFINITIONS "-Minform=inform")
  if(C_COMPILER_HAS_FLAG)
      set(ADD_CFLAGS "${ADD_CFLAGS} -Minform=inform")
  endif()
endif()

set(ADD_CFLAGS "${ADD_CFLAGS} -D_FILE_OFFSET_BITS=64")

if(${OPENSSL_FOUND})
	add_library(fti.static STATIC ${SRC_FTI} ${OPENSSL_LIBRARIES}
    		$<TARGET_OBJECTS:iniparser> $<TARGET_OBJECTS:jerasure>)
	add_library(fti.shared SHARED ${SRC_FTI} ${OPENSSL_LIBRARIES}
    		$<TARGET_OBJECTS:iniparser> $<TARGET_OBJECTS:jerasure>)
    set(ADD_CFLAGS "${ADD_CFLAGS} -DHAVE_OPENSSL")
else()
	add_library(fti.static STATIC ${SRC_FTI}
	    $<TARGET_OBJECTS:iniparser> $<TARGET_OBJECTS:jerasure>
		$<TARGET_OBJECTS:md5>)
	add_library(fti.shared SHARED ${SRC_FTI}
	    $<TARGET_OBJECTS:iniparser> $<TARGET_OBJECTS:jerasure>
		$<TARGET_OBJECTS:md5>)
	append_property(TARGET fti.static fti.shared
		PROPERTY LINK_FLAGS "-DMD5P=TRUE")
	unset(OPENSSL_LIBRARIES)
endif()
set_target_properties(fti.static fti.shared PROPERTIES POSITION_INDEPENDENT_CODE True)

append_property(SOURCE ${SRC_FTI}
    PROPERTY COMPILE_FLAGS "${MPI_C_COMPILE_FLAGS}" "${SIONLIB_CFLAGS}" "${ADD_CFLAGS}")


append_property(TARGET fti.static fti.shared
	PROPERTY LINK_FLAGS "${MPI_C_LINK_FLAGS}")
set_property(TARGET fti.static fti.shared
	PROPERTY OUTPUT_NAME fti)

#PGCC C and C++ use builtin math functions, which are much more efficient than library calls.
#http://www.cecalc.ula.ve/documentacion/tutoriales/HPF/pgiws_ug/pgi30u09.htm
if(NOT "$ENV{COMPILER}" STREQUAL "pgi")
	find_library(LIBM m DOC "The math library")
endif()

if(ZLIB_FOUND)
    target_link_libraries(fti.static ${MPI_C_LIBRARIES} "${LIBM}" "${OPENSSL_LIBRARIES}" "${ZLIB_LIBRARIES}")
    target_link_libraries(fti.shared ${MPI_C_LIBRARIES} "${LIBM}" "${OPENSSL_LIBRARIES}" "${ZLIB_LIBRARIES}")
else()
    target_link_libraries(fti.static ${MPI_C_LIBRARIES} "${LIBM}" "${OPENSSL_LIBRARIES}")
    target_link_libraries(fti.shared ${MPI_C_LIBRARIES} "${LIBM}" "${OPENSSL_LIBRARIES}")
endif()

if(ENABLE_LUSTRE)
    if(LUSTREAPI_FOUND)
        include_directories(${LUSTREAPI_INCLUDE_DIRS})
        append_property(SOURCE ${SRC_FTI} PROPERTY COMPILE_DEFINITIONS LUSTRE)
        target_link_libraries(fti.static ${LUSTREAPI_LIBRARIES})
        target_link_libraries(fti.shared ${LUSTREAPI_LIBRARIES})
    else()
        message(WARNING "
  ** Lustre could not be found!
  *  You may specify:
  *  -DLUSTREAPI_CMAKE_LIBRARY_DIRS:PATH=<path to liblustreapi.a>
  *  and
  *  -DLUSTREAPI_CMAKE_INCLUDE_DIRS:PATH=<path to liblustreapi.h>")
    endif()
endif()

if(ENABLE_SIONLIB)
    set(SIONLIBBASE "" CACHE FILEPATH "base path to SIONlib installation")
    set(SIONLIB_INCLUDE_DIR "${SIONLIBBASE}/include/")
    include_directories("${SIONLIB_INCLUDE_DIR}")
    set(SIONLIB_CFLAGS "-I${SIONLIB_INCLUDE_DIR} -D_SION_LINUX -DSION_MPI")
    find_library(SIONLIB_MPI NAMES "sionmpi_64" PATHS ${SIONLIBBASE} PATH_SUFFIXES "lib" NO_DEFAULT_PATH)
    find_library(SIONLIB_GEN NAMES "siongen_64" PATHS ${SIONLIBBASE} PATH_SUFFIXES "lib" NO_DEFAULT_PATH)
    find_library(SIONLIB_SER NAMES "sionser_64" PATHS ${SIONLIBBASE} PATH_SUFFIXES "lib" NO_DEFAULT_PATH)
    find_library(SIONLIB_COM NAMES "sioncom_64" PATHS ${SIONLIBBASE} PATH_SUFFIXES "lib" NO_DEFAULT_PATH)
    find_library(SIONLIB_COM_LOCK NAMES "sioncom_64_lock_none" PATHS ${SIONLIBBASE} PATH_SUFFIXES "lib" NO_DEFAULT_PATH)
    target_link_libraries(fti.static "${SIONLIB_MPI}" "${SIONLIB_GEN}" "${SIONLIB_SER}" "${SIONLIB_COM}" "${SIONLIB_COM_LOCK}")
    target_link_libraries(fti.shared "${SIONLIB_MPI}" "${SIONLIB_GEN}" "${SIONLIB_SER}" "${SIONLIB_COM}" "${SIONLIB_COM_LOCK}")
    add_definitions(-DENABLE_SIONLIB)
endif()

if(ENABLE_HDF5)
    find_package(HDF5 COMPONENTS HL C REQUIRED)
	if(HDF5_FOUND)
		add_definitions(-DENABLE_HDF5)
		include_directories(${HDF5_INCLUDE_DIRS})
		append_property(SOURCE ${SRC_FTI} PROPERTY COMPILE_DEFINITIONS HDF5)
		target_link_libraries(fti.static ${HDF5_LIBRARIES})
		target_link_libraries(fti.shared ${HDF5_LIBRARIES})
		target_link_libraries(fti.static ${HDF5_HL_LIBRARIES})
		target_link_libraries(fti.shared ${HDF5_HL_LIBRARIES})
	else()
		message(WARNING "
		** HDF5 could not be found!
		*  You may specify:
		*  -DHDF5_ROOT as the path to the HDF5 installation to use.
		**")
	endif()
endif()

set(FTI_TARGETS fti.static fti.shared)
install(TARGETS fti.static fti.shared DESTINATION lib)
install(FILES "include/fti.h"
	DESTINATION include)

if(ENABLE_FORTRAN)
    CHECK_Fortran_COMPILER_FLAG("-cpp" FC_HAS_CPP_FLAG)
    if(NOT FC_HAS_CPP_FLAG)
        message(FATAL_ERROR "
  ** Fortran compiler does not support -cpp flag.
  *  Please use a newer version!")
    endif()
    try_compile(MPI_USE_MOD ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/test/cmake/checkMPImodule.f90
	    LINK_LIBRARIES ${MPI_Fortran_LIBRARIES}
	    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${MPI_Fortran_INCLUDE_PATH}")
	if(NOT MPI_USE_MOD)
	    message(WARNING "
  ** mpi.mod was generated by a different compiler/compiler-version
  *  The Fortran examples and tests will be build using 'include mpif.h'!")

	endif()

    add_subdirectory(vendor/bpp/ bpp/ EXCLUDE_FROM_ALL)

	bpp_preprocess(BPP_FTI_F90
		src/fortran/interface.F90.bpp)

	add_custom_target(bpp_file DEPENDS "${BPP_FTI_F90}") # to serialize src generation

	set(SRC_FTI_F90 ${BPP_FTI_F90}
		src/fortran/ftif.c)
	append_property(SOURCE ${SRC_FTI_F90}
		PROPERTY COMPILE_FLAGS "${MPI_Fortran_COMPILE_FLAGS}")

	add_library(fti_f90.static STATIC ${SRC_FTI_F90})
	add_dependencies(fti_f90.static bpp_file) # to serialize src generation
	target_link_libraries(fti_f90.static
		fti.static ${MPI_Fortran_LIBRARIES} ${MPI_C_LIBRARIES})
	set_property(TARGET fti_f90.static PROPERTY Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include")

	add_library(fti_f90.shared SHARED ${SRC_FTI_F90})
	add_dependencies(fti_f90.shared bpp_file) # to serialize src generation
	add_dependencies(fti_f90.shared fti_f90.static) # to serialize mod generation
	target_link_libraries(fti_f90.shared
		fti.shared ${MPI_Fortran_LIBRARIES} ${MPI_C_LIBRARIES})

	append_property(TARGET fti_f90.static fti_f90.shared
		PROPERTY LINK_FLAGS "${MPI_Fortran_LINK_FLAGS} ${MPI_C_LINK_FLAGS}")
	set_property(TARGET fti_f90.static fti_f90.shared
		PROPERTY OUTPUT_NAME fti_f90)

	list(APPEND FTI_TARGETS fti_f90.static fti_f90.shared)
	install(TARGETS fti_f90.static fti_f90.shared
		DESTINATION lib)
	install(FILES ${CMAKE_Fortran_MODULE_DIRECTORY}/fti.mod
		DESTINATION include)
	set_target_properties(fti_f90.static fti_f90.shared PROPERTIES POSITION_INDEPENDENT_CODE True)
endif()

if(ENABLE_EXAMPLES)
	add_subdirectory(examples)
endif()

if(ENABLE_TESTS)
	add_subdirectory(test)
endif()

if(ENABLE_DOCU)
    add_subdirectory(doc/Doxygen)
endif()

# installable config

set(WORK_DIR "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}")
set(FTI_INCLUDE_PATH "${CMAKE_INSTALL_PREFIX}/include")
install(TARGETS ${FTI_TARGETS} DESTINATION lib/CMake/FTI)
configure_file(CMakeScripts/FTIConfig.cmake.in "${WORK_DIR}/FTIConfig.cmake" @ONLY)
export(PACKAGE FTI)
install(FILES "${WORK_DIR}/FTIConfig.cmake" DESTINATION lib/CMake/FTI)


# in tree config

set(FTI_INCLUDE_PATH "${CMAKE_CURRENT_BINARY_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE STRING "FTI Include path")
export(TARGETS ${FTI_TARGETS} FILE "${PROJECT_BINARY_DIR}/FTILib.cmake")
configure_file(CMakeScripts/FTIConfig.cmake.in "${PROJECT_BINARY_DIR}/FTIConfig.cmake" @ONLY)

# set up exaples in build/examples/

file(COPY examples/config.fti DESTINATION examples)
file(COPY examples/configBkp.fti DESTINATION examples)
file(COPY examples/plot.sh DESTINATION examples)
file(COPY examples/vplot.plg DESTINATION examples)
file(COPY examples/README DESTINATION examples)
