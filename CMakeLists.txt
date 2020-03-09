cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(FTI LANGUAGES C)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeScripts")
include(GNUInstallDirs)
include(CheckCCompilerFlag)

option(BUILD_SHARED_LIBS "Build shared libraries rather than static ones" ON)
option(ENABLE_FORTRAN    "Enables the generation of the Fortran wrapper for FTI" OFF)
option(ENABLE_EXAMPLES   "Enables the generation of examples" ON)
option(ENABLE_SIONLIB    "Enables the parallel I/O SIONlib for FTI" OFF)
option(ENABLE_HDF5       "Enables the HDF5 checkpoints for FTI" OFF)
option(ENABLE_TESTS      "Enables the generation of tests" ON)
option(ENABLE_FI_IO      "Enables the I/O failure injection mechanism" OFF)
option(ENABLE_LUSTRE     "Enables Lustre Support" OFF)
option(ENABLE_DOCU       "Enables the generation of a Doxygen documentation" OFF)
option(ENABLE_TUTORIAL   "Enables the generation of tutorial files" OFF)
option(NO_OPENSSL        "Disable the use of OpenSSL" OFF)


## Enable optional languages

if("${ENABLE_GPU}")
    enable_language(CUDA CXX)
endif()

if("${ENABLE_FORTRAN}")
    enable_language(Fortran)
    CheckCFortranMatch()
endif()

## Find dependencies

find_package(MPI REQUIRED)

add_subdirectory(deps/iniparser)

add_subdirectory(deps/jerasure)

#TODO: this should be required with a ENABLE_ZLIB flag
find_package(ZLIB)

if("${NO_OPENSSL}")
    string(ASCII 27 Esc)
    set(ColourReset "${Esc}[m")
    set(Red         "${Esc}[31m")
    message(STATUS "${Red}Using slower build in MD5 algorithm.${ColourReset}")
    add_subdirectory(deps/md5)
else()
    find_package(OPENSSL REQUIRED)
endif()

if("${ENABLE_LUSTRE}")
    #TODO: this should be required
    find_package(LUSTREAPI)
    if(NOT "${LUSTREAPI_FOUND}")
        message(WARNING "
  ** Lustre could not be found!
  *  You may specify:
  *  -DLUSTREAPI_CMAKE_LIBRARY_DIRS:PATH=<path to liblustreapi.a>
  *  and
  *  -DLUSTREAPI_CMAKE_INCLUDE_DIRS:PATH=<path to liblustreapi.h>")
    endif()
endif()

if("${ENABLE_HDF5}")
    find_package(HDF5 COMPONENTS HL C REQUIRED)
endif()


## Create the C library

add_library(fti
    src/postckpt.c
    src/conf.c
    src/fti-io.c
    src/recover.c
    src/postreco.c
    src/failure-injection.c
    src/api-cuda.c
    src/api.c
    src/checkpoint.c
    src/dcp.c
    src/IO/posix-dcp.c
    src/IO/hdf5-fti.c
    src/IO/ftiff.c
    src/IO/mpio.c
    src/IO/posix.c
    src/IO/ftiff-dcp.c
    src/macros.c
    src/stage.c
    src/meta.c
    src/icp.c
    src/fortran/ftif.c
    src/topo.c
    src/tools.c
    src/utility.c
)

target_include_directories(fti  PUBLIC
    "$<BUILD_INTERFACE:${FTI_BINARY_DIR}>" 
    "$<BUILD_INTERFACE:${FTI_SOURCE_DIR}/include/>"
    PRIVATE
    "$<BUILD_INTERFACE:${FTI_SOURCE_DIR}/src>"
)

set_property(TARGET fti PROPERTY POSITION_INDEPENDENT_CODE TRUE)

if("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
    target_compile_options(fti PRIVATE "-Wall")
elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(fti PRIVATE "-Wall")
    check_c_compiler_flag("-Wno-format-truncation" C_COMPILER_HAS_FLAG)
    if(C_COMPILER_HAS_FLAG)
        target_compile_options(fti PRIVATE "-Wno-format-truncation")
    endif()
elseif("${CMAKE_C_COMPILER_ID}" STREQUAL "PGI")
    check_c_compiler_flag("-Minform=inform" C_COMPILER_HAS_FLAG)
    if(C_COMPILER_HAS_FLAG)
        target_compile_options(fti PRIVATE "-Minform=inform")
    endif()
endif()

if("${ENABLE_GPU}")
    target_sources(fti PRIVATE src/IO/cuda-md5/md5Opt.cu)
    target_compile_definitions(fti PRIVATE "GPUSUPPORT")
else()
    target_sources(fti PRIVATE src/IO/cuda-md5/md5Opt.c)
endif()

link_libraries(fti PRIVATE
    iniparser
    jerasure
    MPI::MPI_C
)

#PGCC C and C++ use builtin math functions, which are much more efficient than library calls.
#http://www.cecalc.ula.ve/documentacion/tutoriales/HPF/pgiws_ug/pgi30u09.htm
if(NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "PGI")
    find_library(LIBM m DOC "The math library")
    target_link_libraries(fti PRIVATE "${LIBM}")
endif()

if("${OPENSSL_FOUND}")
    link_libraries(fti PRIVATE OpenSSL::SSL)
    set(HAVE_OPENSSL 1)
else()
    link_libraries(fti PRIVATE md5)
	target_compile_definitions(fti PRIVATE "MD5P=TRUE")
    set(HAVE_OPENSSL 0)
endif()
configure_file(deps/md5/md5.h.in "${CMAKE_CURRENT_SOURCE_DIR}/deps/md5/md5.h" @ONLY)

if("${ZLIB_FOUND}")
    link_libraries(fti PRIVATE ZLIB::ZLIB)
else()
    target_compile_definitions(fti PRIVATE "FTI_NOZLIB")
endif()

if("${LUSTREAPI_FOUND}")
    target_link_libraries(fti PRIVATE LUSTREAPI::lustre)
    target_compile_definitions(fti PRIVATE "LUSTRE")
endif()

if("${SIONLIB_FOUND}")
    target_sources(fti PRIVATE src/IO/sion-fti.c)
endif()


if(ENABLE_HDF5)
    add_definitions(-DENABLE_HDF5)
    include_directories(${HDF5_INCLUDE_DIRS})
    append_property(SOURCE ${SRC_FTI} PROPERTY COMPILE_DEFINITIONS HDF5)
    target_link_libraries(fti.static ${HDF5_LIBRARIES})
    target_link_libraries(fti.shared ${HDF5_LIBRARIES})
    target_link_libraries(fti.static ${HDF5_HL_LIBRARIES})
    target_link_libraries(fti.shared ${HDF5_HL_LIBRARIES})
endif()

add_library(FTI::fti ALIAS fti)


# add compiler flags


if(ENABLE_SIONLIB)
    set(SIONLIBBASE "" CACHE FILEPATH "base path to SIONlib installation")
    set(SIONLIB_INCLUDE_DIR "${SIONLIBBASE}/include/")
    include_directories("${SIONLIB_INCLUDE_DIR}")
    set(SIONLIB_CFLAGS "-I${SIONLIB_INCLUDE_DIR} -DSION_DEBUG -D_SION_LINUX  -DSION_MPI")
    find_library(SIONLIB_MPI NAMES "sionmpi_64" PATHS ${SIONLIBBASE} PATH_SUFFIXES "lib" NO_DEFAULT_PATH)
    find_library(SIONLIB_GEN NAMES "siongen_64" PATHS ${SIONLIBBASE} PATH_SUFFIXES "lib" NO_DEFAULT_PATH)
    find_library(SIONLIB_SER NAMES "sionser_64" PATHS ${SIONLIBBASE} PATH_SUFFIXES "lib" NO_DEFAULT_PATH)
    find_library(SIONLIB_COM NAMES "sioncom_64" PATHS ${SIONLIBBASE} PATH_SUFFIXES "lib" NO_DEFAULT_PATH)
    find_library(SIONLIB_COM_LOCK NAMES "sioncom_64_lock_none" PATHS ${SIONLIBBASE} PATH_SUFFIXES "lib" NO_DEFAULT_PATH)
    target_link_libraries(fti.static "${SIONLIB_MPI}" "${SIONLIB_GEN}" "${SIONLIB_SER}" "${SIONLIB_COM}" "${SIONLIB_COM_LOCK}")
    target_link_libraries(fti.shared "${SIONLIB_MPI}" "${SIONLIB_GEN}" "${SIONLIB_SER}" "${SIONLIB_COM}" "${SIONLIB_COM_LOCK}")
    add_definitions(-DENABLE_SIONLIB)
endif()

if(ENABLE_FI_IO)
    add_definitions(-DENABLE_FTI_FI_IO)
endif()

append_property(SOURCE ${SRC_FTI}
    PROPERTY COMPILE_FLAGS "${MPI_C_COMPILE_FLAGS} ${SIONLIB_CFLAGS} ${ADD_CFLAGS}")

set(FTI_TARGETS fti.static fti.shared)
install(TARGETS fti.static fti.shared DESTINATION lib)
install(FILES "include/fti.h"
	DESTINATION include)
install(FILES "include/fti-intern.h"
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

if(ENABLE_TUTORIAL)
	add_subdirectory(tutorial)
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

set(INSTALL_FTIDATADIR "${CMAKE_INSTALL_DATADIR}/fti" CACHE PATH "FTI data directory (DATADIR/fti)")
configure_file(scripts/env.bash.in env.bash @ONLY)
configure_file(cmake/FTIConfig.cmake.in FTIConfig.cmake @ONLY)
install(FILES "${FTI_BINARY_DIR}/env.bash"
		DESTINATION "${INSTALL_FTIDATADIR}"
		COMPONENT Runtime
)

set(INSTALL_CMAKEDIR "${INSTALL_FTIDATADIR}/cmake" CACHE PATH "cmake modules (FTIDATADIR/cmake)")
write_basic_package_version_file("${FTI_BINARY_DIR}/FTIConfigVersion.cmake"
		VERSION "${FTI_VERSION}"
		COMPATIBILITY AnyNewerVersion
)
install(EXPORT FTI_EXPORT NAMESPACE "FTI::" DESTINATION "${INSTALL_CMAKEDIR}" FILE "fti.cmake")
install(FILES
		cmake/CMakeFindDependencyMacro.cmake
		cmake/FindPackageHandleStandardArgs.cmake
		cmake/FindPackageMessage.cmake
		"${FTI_BINARY_DIR}/FTIConfig.cmake"
		"${FTI_BINARY_DIR}/FTIConfigVersion.cmake"
		DESTINATION "${INSTALL_CMAKEDIR}"
		COMPONENT Development
)

# set up examples in build/examples/
file(COPY examples/config.fti DESTINATION examples)
file(COPY examples/configBkp.fti DESTINATION examples)
file(COPY examples/plot.sh DESTINATION examples)
file(COPY examples/vplot.plg DESTINATION examples)
file(COPY examples/README DESTINATION examples)
