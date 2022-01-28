cmake_minimum_required(VERSION 3.4 FATAL_ERROR)

if(POLICY CMP0054)
	cmake_policy(SET CMP0054 NEW)
endif()

# ---------------------------------- Options ----------------------------------

option(ENABLE_FORTRAN "Enables the build of a Fortran wrapper for FTI" OFF)
option(ENABLE_LUSTRE "Enables Lustre Support" OFF)
option(ENABLE_OPENSSL "Enables linking against system OpenSSL library" ON)
# Additional IO Modes
option(ENABLE_SIONLIB "Enables the parallel I/O SIONlib for FTI" OFF)
option(ENABLE_HDF5 "Enables the HDF5 checkpoints for FTI" OFF)
option(ENABLE_IME_NATIVE "Enables the IME native API" OFF)
# User Extra Utilities
option(ENABLE_EXAMPLES "Enables the generation of examples" ON)
option(ENABLE_TUTORIAL "Enables the generation of tutorial files" OFF)
# Developer Extra Utilities
option(ENABLE_FI_IO "Enables the I/O failure injection mechanism" OFF)
option(ENABLE_DOCU "Enables the generation of a Doxygen documentation" OFF)
option(ENABLE_TESTS "Enables the generation of tests" OFF)
option(ENABLE_COVERAGE "Enable coverage metrics for FTI" OFF)

# ----------------------------- Project Definition ----------------------------

set(LANGUAGES "C" "CXX")
if(ENABLE_FORTRAN)
  list(APPEND LANGUAGES "Fortran")
endif()
if(ENABLE_GPU)
  list(APPEND LANGUAGES "CUDA" "CXX")
endif()

project("FTI" VERSION 1.6 LANGUAGES ${LANGUAGES})

# ----------------------- Global Properties and Set-up ------------------------

set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS ON)

include(GNUInstallDirs)
include(CheckCCompilerFlag)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeScripts")

set(TOOLS_DIR "${CMAKE_SOURCE_DIR}/testing/tools/cmake")

# Include directories necessary to process FTI headers
set(INC "")
# Include directories necessary to process FTI headers and source
set(INC_PUB "")
# Include directories necessary to process FTI source
set(INC_PRIV "")

set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include)

# ----------------------------- Find Dependencies -----------------------------

# Search dependencies in following order: required, recommended, optional

# Package: MPI (Required)
find_package(MPI REQUIRED)

# Package: OpenSSL (Recommended)
set(HAVE_OPENSSL 0)
if(ENABLE_OPENSSL)
    find_package(OPENSSL)
    if(OPENSSL_FOUND)
        set(HAVE_OPENSSL 1)
        list(APPEND INC_PRIV ${OPENSSL_INCLUDE_DIR})
    endif()
endif()

# Package: ZLIB (Recommended)
find_package(ZLIB)

# Library: LibM (Conditional)
# PGCC C and C++ use builtin math functions.
# These are much more efficient than library calls.
# http://www.cecalc.ula.ve/documentacion/tutoriales/HPF/pgiws_ug/pgi30u09.htm
if(NOT "$ENV{COMPILER}" STREQUAL "pgi")
	find_library(LIBM m DOC "The math library")
endif()

# Package: CUDA (Optional)
if(ENABLE_GPU)
    FIND_PACKAGE(CUDA)
endif()

# Package: Lustre (Optional)
if(ENABLE_LUSTRE)
    find_package(LUSTREAPI)
    if(NOT LUSTREAPI_FOUND)
        message(WARNING "Lustre could not be found!\n"
            "You may specify:\n"
            "-DLUSTREAPI_CMAKE_LIBRARY_DIRS:PATH=<path to liblustreapi.a>"
            " and\n"
            "-DLUSTREAPI_CMAKE_INCLUDE_DIRS:PATH=<path to liblustreapi.h>")
    endif()
endif()

# Library: HDF5 (Optional)
if(ENABLE_HDF5)
    set(HDF5_PREFER_PARALLEL ON)
    find_package(HDF5 COMPONENTS HL C REQUIRED)
    if(NOT HDF5_IS_PARALLEL)
        message(FATAL_ERROR "Serial HDF5 found but HDF5 with MPI needed")
    endif()
endif()

# Library: IME (Optional)
if(ENABLE_IME_NATIVE)
    if (IMELIBBASE)
        set(IMELIBBASE "" CACHE FILEPATH
            "base path to IME client lib installation")
    else()
        set(IMELIBBASE "/opt/ddn/ime/" CACHE FILEPATH
            "base path to IME client lib installation" FORCE)
    endif()
    set(IMELIB_INCLUDE_DIR "${IMELIBBASE}/include/")
    find_library(IMELIB NAMES "im_client" PATHS ${IMELIBBASE}
                PATH_SUFFIXES "lib" NO_DEFAULT_PATH)
endif()

# Library: SIONLib (Optional)
if(ENABLE_SIONLIB)
    set(SIONLIBBASE "" CACHE FILEPATH "base path to SIONlib installation")
    set(SIONLIB_INCLUDE_DIR "${SIONLIBBASE}/include/")

    list(APPEND INC_PRIV ${SIONLIB_INCLUDE_DIR})
    find_library(SIONLIB_MPI NAMES "sionmpi_64" 
                PATHS ${SIONLIBBASE} PATH_SUFFIXES "lib" NO_DEFAULT_PATH)
    find_library(SIONLIB_GEN NAMES "siongen_64"
                PATHS ${SIONLIBBASE} PATH_SUFFIXES "lib" NO_DEFAULT_PATH)
    find_library(SIONLIB_SER NAMES "sionser_64"
                PATHS ${SIONLIBBASE} PATH_SUFFIXES "lib" NO_DEFAULT_PATH)
    find_library(SIONLIB_COM NAMES "sioncom_64"
                PATHS ${SIONLIBBASE} PATH_SUFFIXES "lib" NO_DEFAULT_PATH)
    find_library(SIONLIB_COM_LOCK NAMES "sioncom_64_lock_none"
                PATHS ${SIONLIBBASE} PATH_SUFFIXES "lib" NO_DEFAULT_PATH)
endif()

# --------------------------- Fortran Support Checks --------------------------

if(ENABLE_FORTRAN)
    # 1) FTI requires the -cpp flag for the creation of Fortran bindings
    set(CMAKE_Fortran_FLAGS "-cpp")
    # 2) Compilers must share the major version or risk undefined behavior
    set(F_CC_MAJOR 0)
    # 3) FTI will ideally use the Fortran MPI.mod if compatible with compiler
    set(MPI_USE_MOD 1)

    # Load modules to perform the aforementioned checks
    include(FortranCInterface)
    include(CheckFortranCompilerFlag)

    # 1) Check if the Fortran compiler supports the -cpp flag
    CHECK_Fortran_COMPILER_FLAG("-cpp" FC_HAS_CPP_FLAG)
    if(NOT FC_HAS_CPP_FLAG)
        message(FATAL_ERROR
        "Fortran compiler does not support -cpp flag,"
        " please use a newer version!")
    endif()

    # 2) Determine if C and Fortran compilers are of the same major version
    # 2.1) Determine the C compiler major version
    string(REGEX MATCH "[0-9]+" C_CC_MAJOR ${CMAKE_C_COMPILER_VERSION})
    # 2.2) Determine the Fortran compiler major version
    string(REGEX MATCH "[0-9]+" F_CC_MAJOR ${CMAKE_Fortran_COMPILER_VERSION})
    # 2.2.a) If fail to automatically detect the version, try manually
    if (NOT F_CC_MAJOR)
        message(AUTHOR_WARNING
        "Cmake variable 'CMAKE_Fortran_COMPILER_VERSION' is undefined."
        " Attempting to determine it manually...")

        # 2.2.b) Compare compiler name to Intel, gfortran and PGI
        string(COMPARE EQUAL ${CMAKE_Fortran_COMPILER_ID} "Intel" F_CC_INTEL)
        string(COMPARE EQUAL ${CMAKE_Fortran_COMPILER_ID} "GNU" F_CC_GNU)
        string(COMPARE EQUAL ${CMAKE_Fortran_COMPILER_ID} "PGI" F_CC_PGI)

        # 2.2.c) Select a utility program to output the compiler version
        if(${F_CC_INTEL})
            set(F_CC_CHECK "${TOOLS_DIR}/intel_major_ver.f90")
        elseif(${F_CC_GNU})
            set(F_CC_CHECK "${TOOLS_DIR}/gnu_major_ver.f90")
        elseif(${F_CC_PGI})
            set(F_CC_CHECK "${TOOLS_DIR}/pgi_major_ver.f90")
        else()
            message(FATAL_ERROR
                    "Could not determine the Fortran compiler version")
        endif()

        # 2.2.d) Compile, execute and store output of utility program
        try_run(PROG_RAN COMPILE_SUCCESS "${CMAKE_BINARY_DIR}" "${F_CC_CHECK}"
                RUN_OUTPUT_VARIABLE OUT)
        if(COMPILE_SUCCESS and PROG_RAN)
            string(REGEX MATCH "[0-9]+" F_CC_MAJOR "${OUT}")
        endif()
        # 2.2.e) Verify if successful in manual detection
        if(F_CC_MAJOR)
            message(AUTHOR_WARNING
                    "The Fortran compiler version is ${F_CC_MAJOR}")
        else()
            message(FATAL_ERROR
                    "Could not determine the Fortran compiler version")
        endif()
    endif()

    # 2.2.f) Check if both compilers are of the same type/version
    set(C_CC_ID "${CMAKE_C_COMPILER_ID}_${C_CC_MAJOR}")
    set(F_CC_ID "${CMAKE_Fortran_COMPILER_ID}_${F_CC_MAJOR}")
    string(COMPARE NOTEQUAL ${C_CC_ID} ${F_CC_ID} COMPILER_VER_DIFFER)
    if(${COMPILER_VER_DIFFER})
        message(WARNING
                "Different compiler idetifications for Fortran and C!")
    endif()
    
    # 2.2.f) Check if c/c++ and fortran compilers are compatible
    FortranCInterface_VERIFY()
    FortranCInterface_VERIFY(CXX)
    if((NOT ${FortranCInterface_VERIFIED_C}) AND (NOT ${FortranCInterface_VERIFIED_CXX}))
        message(FATAL_ERROR
                "Different compiler idetifications for Fortran and C!")
    endif()

    # 3) Determine if using MPI.mod or mpif.h
    try_compile(MPI_USE_MOD ${CMAKE_BINARY_DIR}
        ${TOOLS_DIR}/checkMPImodule.f90
        LINK_LIBRARIES ${MPI_Fortran_LIBRARIES}
	    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${MPI_Fortran_INCLUDE_PATH}")
	if(NOT MPI_USE_MOD)
	    message(WARNING
            "MPI.mod was generated by a different compiler/compiler-version."
            " Fortran binaries will be built with 'mpif.h'")
	endif()
endif()

# ------------------ Definition of source and include files -------------------

# Define unconditional FTI sources
set(SRC_FTI
    src/fortran/ftif.c
    src/util/tools.c
    src/util/utility.c
    src/util/keymap.c
    src/util/dataset.c
    src/util/ini.c
    src/util/macros.c
    src/util/failure-injection.c
    src/util/metaqueue.c
    src/IO/posix-dcp.c
    src/IO/hdf5-fti.c
    src/IO/ftiff.c
    src/IO/mpio.c
    src/IO/posix.c
    src/IO/ftiff-dcp.c
    src/postckpt.c
    src/conf.c
    src/fti-io.c
    src/recover.c
    src/postreco.c
    src/api-cuda.c
    src/api.c
    src/checkpoint.c
    src/dcp.c
    src/stage.c
    src/meta.c
    src/icp.c
    src/topo.c
)

# FTI Dependencies
add_subdirectory(src/deps)
list(APPEND SRC_FTI $<TARGET_OBJECTS:iniparser> $<TARGET_OBJECTS:jerasure>)

# --- Define FTI conditional sources in alphabetical order ---

# CUDA
if (ENABLE_GPU)
    list(APPEND SRC_FTI "src/IO/cuda-md5/md5Opt.cu")
else ()
    list(APPEND SRC_FTI "src/IO/cuda-md5/md5Opt.c")
endif()

# IME
if (ENABLE_IME_NATIVE)
    list(APPEND SRC_FTI "src/IO/ime.c")
endif()

# SIONLib IO Mode
if (ENABLE_SIONLIB)
    list(APPEND SRC_FTI "src/IO/sion-fti.c")
endif()

# OpenSSL
if(HAVE_OPENSSL)
    list(APPEND SRC_FTI ${OPENSSL_LIBRARIES})
else()
    list(APPEND SRC_FTI $<TARGET_OBJECTS:md5>)
	unset(OPENSSL_LIBRARIES)
endif()

# ------------------------- Definition of FTI targets -------------------------

# FTI main targets
set(fti_targets "fti.static" "fti.shared")
add_library("fti.static" STATIC ${SRC_FTI})
add_library("fti.shared" SHARED ${SRC_FTI})

# Base properties
set_property(TARGET ${fti_targets} PROPERTY OUTPUT_NAME "fti")
set_property(TARGET ${fti_targets} PROPERTY POSITION_INDEPENDENT_CODE True)
set_property(TARGET ${fti_targets} PROPERTY LINK_FLAGS
            "${MPI_C_LINK_FLAGS}")

# Base include directories
list(APPEND INC_PRIV
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
list(APPEND INC_PUB
    ${MPI_C_INCLUDE_PATH}
    ${MPI_Fortran_INCLUDE_PATH}
)

# ------------------------- Flags, Links and Includes -------------------------

# Link utility function
function(link_to_fti)
    foreach(fti_target ${fti_targets})
        target_link_libraries(${fti_target} ${ARGN})
    endforeach(fti_target)
endfunction(link_to_fti)

# Compilation flags utility function
function(ADD_CFLAG_IF_EXISTS CFLAG)
  try_compile(HAS_FLAG ${CMAKE_BINARY_DIR} ${TOOLS_DIR}/minimal.c
            COMPILE_DEFINITIONS "${CFLAG}")
  if(HAS_FLAG)
      set(ADD_CFLAGS "${ADD_CFLAGS} ${CFLAG}" PARENT_SCOPE)
  endif()
endfunction()

# Unconditional definitions
set(ADD_CFLAGS "-D_FILE_OFFSET_BITS=64")
link_to_fti(${MPI_C_LIBRARIES} ${LIBM} ${OPENSSL_LIBRARIES} ${CUDA_LIBRARIES})

# --- Compiler Flags definitions ---

# GCC compiler flags
if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    ADD_CFLAG_IF_EXISTS("-Wno-format-truncation")
endif()

# PGI
if("${CMAKE_C_COMPILER_ID}" STREQUAL "PGI")
    ADD_CFLAG_IF_EXISTS("-Minform=inform")
else()
    set(ADD_CFLAGS "${ADD_CFLAGS} -Wall")
endif()

# --- Conditional definitions in alphabetical order ---

# Coverage
if(ENABLE_COVERAGE)
    set(ADD_CFLAGS "${ADD_CFLAGS} --coverage -O0 -g")
    link_to_fti("gcov" "--coverage")
endif()

# CUDA
if (ENABLE_GPU)
  set(ADD_CFLAGS "${ADD_CFLAGS} -DGPUSUPPORT")
  list(APPEND INC_PUB ${CUDA_INCLUDE_DIRS})
endif()

# FI-IO
if(ENABLE_FI_IO)
    set(ADD_CFLAGS "${ADD_CFLAGS} -DENABLE_FTI_FI_IO")
endif()

# HDF5
if(ENABLE_HDF5)
		set(_COMPILE_DEFS "${_COMPILE_DEFS} ENABLE_HDF5")
    set(ADD_CFLAGS "${ADD_CFLAGS} -DENABLE_HDF5 -DHDF5")
    list(APPEND INC_PUB ${HDF5_INCLUDE_DIRS})
	link_to_fti(${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})
endif()

# IME
if(ENABLE_IME_NATIVE)
    set(ADD_CFLAGS "${ADD_CFLAGS} -DENABLE_IME_NATIVE")
    list(APPEND INC_PUB ${IMELIB_INCLUDE_DIR})
    link_to_fti(${IMELIB})
endif()

# LUSTRE
if(LUSTREAPI_FOUND)
    set(ADD_CFLAGS "${ADD_CFLAGS} -DLUSTRE")
    list(APPEND INC_PRIV ${LUSTREAPI_INCLUDE_DIRS})
    link_to_fti(${LUSTREAPI_LIBRARIES})
endif()

# OpenSSL
set(ADD_CFLAGS "${ADD_CFLAGS} -DHAVE_OPENSSL=${HAVE_OPENSSL}")

# SIONLib
if(ENABLE_SIONLIB)
    set(ADD_CFLAGS "${ADD_CFLAGS} -DENABLE_SIONLIB -DSION_DEBUG -D_SION_LINUX -DSION_MPI")
    list(APPEND INC_PRIV ${SIONLIB_INCLUDE_DIR})
    link_to_fti(${SIONLIB_MPI} ${SIONLIB_GEN} ${SIONLIB_SER}
                ${SIONLIB_COM} ${SIONLIB_COM_LOCK})
endif()

# ZLib
if(NOT ZLIB_FOUND)
    set(ADD_CFLAGS "${ADD_CFLAGS} -DFTI_NOZLIB")
else()
    link_to_fti(${ZLIB_LIBRARIES})
endif()

# TODO: COMPILER_FLAGS is superseded by COMPILER_OPTIONS (CMake 3.6+)
set_property(SOURCE ${SRC_FTI} PROPERTY COMPILE_FLAGS "${MPI_C_COMPILE_FLAGS} ${ADD_CFLAGS}")

# -------------------------- Instalation definitions --------------------------

foreach(fti_target ${fti_targets})
        target_include_directories(${fti_target} INTERFACE ${INC})
        target_include_directories(${fti_target} PUBLIC ${INC_PUB})
        target_include_directories(${fti_target} PRIVATE ${INC_PRIV})
endforeach(fti_target)

install(TARGETS ${fti_targets} DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    EXPORT FTI_EXPORT
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

install(FILES
    "include/fti.h"
    "include/fti-defs.h"
    "include/fti-intern.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

# ------------------------------ Fortran targets ------------------------------

if(ENABLE_FORTRAN)
    add_subdirectory(src/deps/bpp-0.3.0/ bpp/ EXCLUDE_FROM_ALL)

	bpp_preprocess(BPP_FTI_F90 src/fortran/interface.F90.bpp)

    # to serialize src generation
	add_custom_target(bpp_file DEPENDS "${BPP_FTI_F90}")

	set(SRC_FTI_F90 ${BPP_FTI_F90} src/fortran/ftif.c)
	set_property(SOURCE ${SRC_FTI_F90} APPEND
		        PROPERTY COMPILE_FLAGS "${MPI_Fortran_COMPILE_FLAGS}")

	add_library(fti_f90.static STATIC ${SRC_FTI_F90})
	add_dependencies(fti_f90.static bpp_file) # to serialize src generation
	target_link_libraries(fti_f90.static
		fti.static ${MPI_Fortran_LIBRARIES} ${MPI_C_LIBRARIES})
    set_property(TARGET fti_f90.static PROPERTY 
                Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include")

    add_library(fti_f90.shared SHARED ${SRC_FTI_F90})
    # to serialize src generation
	add_dependencies(fti_f90.shared bpp_file) 
	add_dependencies(fti_f90.shared fti_f90.static)
	target_link_libraries(fti_f90.shared
		                fti.shared ${MPI_Fortran_LIBRARIES} ${MPI_C_LIBRARIES})

	set_property(TARGET fti_f90.static fti_f90.shared APPEND
		PROPERTY LINK_FLAGS "${MPI_Fortran_LINK_FLAGS} ${MPI_C_LINK_FLAGS}")
	set_property(TARGET fti_f90.static fti_f90.shared
        PROPERTY OUTPUT_NAME fti_f90)
    foreach(tgt fti_f90.static fti_f90.shared)
        set_target_properties(${tgt} PROPERTIES POSITION_INDEPENDENT_CODE True)
        target_include_directories(${tgt} INTERFACE ${INC})
        target_include_directories(${tgt} PUBLIC ${INC_PUB})
        target_include_directories(${tgt} PRIVATE ${INC_PRIV})
    endforeach(tgt)

    install(TARGETS fti_f90.static fti_f90.shared EXPORT FTI_EXPORT
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    )
	install(FILES ${CMAKE_Fortran_MODULE_DIRECTORY}/fti.mod
		DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
endif()

# ------------------------------- Extra folders -------------------------------

if(ENABLE_EXAMPLES)
	add_subdirectory(examples)
endif()

if(ENABLE_TESTS)
    enable_testing()
    add_subdirectory(testing)
endif()

if(ENABLE_TUTORIAL)
	add_subdirectory(tutorial)
endif()

if(ENABLE_DOCU)
    add_subdirectory(docs/Doxygen)
endif()

file(COPY examples/template.ini DESTINATION examples)
file(COPY examples/plot.sh DESTINATION examples)
file(COPY examples/vplot.plg DESTINATION examples)
file(COPY examples/README DESTINATION examples)

# ---------------------------- Compiler Definitions ---------------------------

# create fti definition file 'fti-defs.h'

file(WRITE include/fti-defs.h
  "#ifndef __FTI_DEFS__\n"
  "#define __FTI_DEFS__\n"
)
foreach(_DEF ${_COMPILE_DEFS})
  file(APPEND include/fti-defs.h
    "#ifndef ${_DEF}\n"
    "#define ${_DEF}\n"
    "#endif\n"
  )
endforeach()
file(APPEND include/fti-defs.h
  "#endif // __FTI_DEFS__\n"
)

# -------------------------- Package configurations ---------------------------

# Installable config

include(CMakePackageConfigHelpers)
set(INSTALL_CMAKEDIR "${CMAKE_INSTALL_DATADIR}/FTI/cmake" CACHE PATH
    "cmake modules directory (DATADIR/FTI/cmake)")
write_basic_package_version_file("${CMAKE_BINARY_DIR}/FTIConfigVersion.cmake"
		VERSION ${CMAKE_PROJECT_VERSION}
		COMPATIBILITY AnyNewerVersion
)
install(EXPORT FTI_EXPORT DESTINATION "${INSTALL_CMAKEDIR}" FILE "FTILib.cmake")
install(FILES 
	"${CMAKE_BINARY_DIR}/FTIConfigVersion.cmake"
	CMakeScripts/FTIConfig.cmake
	DESTINATION "${INSTALL_CMAKEDIR}"
)

export(EXPORT FTI_EXPORT FILE "${PROJECT_BINARY_DIR}/FTILib.cmake")
configure_file(CMakeScripts/FTIConfig.cmake
                "${PROJECT_BINARY_DIR}/FTIConfig.cmake" COPYONLY)
