cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)

if(POLICY CMP0054)
	cmake_policy(SET CMP0054 NEW)
endif()

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
	project(FTI C Fortran)
endif()

option(ENABLE_FORTRAN "Enables the generation of the Fortran wrapper for FTI" ON)
option(ENABLE_EXAMPLES "Enables the generation of examples" ON) 

set(SIONLIBBASE "" CACHE FILEPATH "base path to SIONlib installation") 

set(SIONLIB_INCLUDE_DIR "${SIONLIBBASE}/install/sionlib_linux_gomp/include/")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeScripts")
include(AppendProperty)
include(FortranCInterface)

find_package(MPI REQUIRED)

add_subdirectory(deps)

include_directories("${CMAKE_CURRENT_BINARY_DIR}"
	"${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${MPI_Fortran_INCLUDE_PATH} ${MPI_C_INCLUDE_PATH} ${SIONLIB_INCLUDE_DIR})

set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include)

set(SRC_FTI
	src/api.c src/checkpoint.c src/conf.c src/meta.c
	src/postckpt.c src/postreco.c src/recover.c
	src/tools.c src/topo.c)

set(SIONLIB_CFLAGS "-I${SIONLIB_INCLUDE_DIR} -DSION_DEBUG -D_SION_LINUX  -DSION_MPI")

append_property(SOURCE ${SRC_FTI}
    PROPERTY COMPILE_FLAGS "${MPI_C_COMPILE_FLAGS}" "${SIONLIB_CFLAGS}")

add_library(fti.static STATIC ${SRC_FTI}
    $<TARGET_OBJECTS:iniparser> $<TARGET_OBJECTS:jerasure>)
add_library(fti.shared SHARED ${SRC_FTI}
    $<TARGET_OBJECTS:iniparser> $<TARGET_OBJECTS:jerasure>)

append_property(TARGET fti.static fti.shared
	PROPERTY LINK_FLAGS "${MPI_C_LINK_FLAGS}")
set_property(TARGET fti.static fti.shared
	PROPERTY OUTPUT_NAME fti)

find_library(LIBM m DOC "The math library")
find_library(SIONLIB_MPI NAMES "sionmpi_64" PATHS ${SIONLIBBASE} PATH_SUFFIXES "install/sionlib_linux_gomp/lib" NO_DEFAULT_PATH)
find_library(SIONLIB_GEN NAMES "siongen_64" PATHS ${SIONLIBBASE} PATH_SUFFIXES "install/sionlib_linux_gomp/lib" NO_DEFAULT_PATH)
find_library(SIONLIB_SER NAMES "sionser_64" PATHS ${SIONLIBBASE} PATH_SUFFIXES "install/sionlib_linux_gomp/lib" NO_DEFAULT_PATH)
find_library(SIONLIB_COM NAMES "sioncom_64" PATHS ${SIONLIBBASE} PATH_SUFFIXES "install/sionlib_linux_gomp/lib" NO_DEFAULT_PATH)
find_library(SIONLIB_COM_LOCK NAMES "sioncom_64_lock_none" PATHS ${SIONLIBBASE} PATH_SUFFIXES "install/sionlib_linux_gomp/lib" NO_DEFAULT_PATH)

target_link_libraries(fti.static ${MPI_C_LIBRARIES} "${LIBM}" "${SIONLIB_MPI}" "${SIONLIB_GEN}" "${SIONLIB_SER}" "${SIONLIB_COM}" "${SIONLIB_COM_LOCK}") 
target_link_libraries(fti.shared ${MPI_C_LIBRARIES} "${LIBM}" "${SIONLIB_MPI}" "${SIONLIB_GEN}" "${SIONLIB_SER}" "${SIONLIB_COM}" "${SIONLIB_COM_LOCK}")

set(FTI_TARGETS fti.static fti.shared)
install(TARGETS fti.static fti.shared DESTINATION lib)
install(FILES "include/fti.h"
	DESTINATION include)

if(ENABLE_FORTRAN)
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
endif()

if(ENABLE_EXAMPLES)
	add_subdirectory(examples)
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
