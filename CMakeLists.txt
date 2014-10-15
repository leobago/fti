cmake_minimum_required(VERSION 2.8)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeScripts")

# protect the project call to workaround a cmake bug
if("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
	project(FTI C Fortran)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(AppendProperty)
include(FortranCInterface)
find_package(MPI REQUIRED)

option(ENABLE_FORTRAN "Enables the generation of the Fortran wrapper for FTI" ON)

include_directories("${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/include" ${MPI_Fortran_INCLUDE_PATH} ${MPI_C_INCLUDE_PATH})

set (LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/lib)

set(SRC_FTI
	src/galois.c
	src/jerasure.c
	src/dictionary.c
	src/iniparser.c
	src/checkpoint.c
	src/postckpt.c
	src/recover.c
	src/postreco.c
	src/topo.c
	src/conf.c
	src/meta.c
	src/tools.c
	src/api.c
)
append_property(SOURCE ${SRC_FTI} PROPERTY COMPILE_FLAGS " ${MPI_C_COMPILE_FLAGS} ")

add_library(fti.static STATIC ${SRC_FTI})
target_link_libraries(fti.static ${MPI_C_LIBRARIES})
add_library(fti.shared SHARED ${SRC_FTI})
target_link_libraries(fti.shared ${MPI_C_LIBRARIES})
append_property(TARGET fti.static fti.shared PROPERTY LINK_FLAGS " ${MPI_C_LINK_FLAGS} ")
set_property(TARGET fti.static fti.shared PROPERTY OUTPUT_NAME fti)

install(TARGETS fti.static fti.shared DESTINATION lib EXPORT FTILib)
install(FILES "include/fti.h" "include/dictionary.h" "include/galois.h" "include/iniparser.h" "include/jerasure.h" DESTINATION include)

if ( ENABLE_FORTRAN )
	add_subdirectory(vendor/bpp/ bpp/ EXCLUDE_FROM_ALL)

	bpp_preprocess(BPP_FTI_F90
		src/interface.F90.bpp
	)
	add_custom_target(bpp_file DEPENDS "${BPP_FTI_F90}") # to serialize src generation

	set(SRC_FTI_F90
		${BPP_FTI_F90}
		src/ftif.c
	)
	append_property(SOURCE ${SRC_FTI_F90}
		PROPERTY COMPILE_FLAGS
		" ${MPI_Fortran_COMPILE_FLAGS} ")

	add_library(fti_f90.static STATIC ${SRC_FTI_F90})
	add_dependencies(fti_f90.static bpp_file) # to serialize src generation
	target_link_libraries(fti_f90.static fti.static ${MPI_Fortran_LIBRARIES} ${MPI_C_LIBRARIES})

	add_library(fti_f90.shared SHARED ${SRC_FTI_F90})
	add_dependencies(fti_f90.shared bpp_file) # to serialize src generation
	target_link_libraries(fti_f90.shared fti.shared ${MPI_Fortran_LIBRARIES} ${MPI_C_LIBRARIES})

	append_property(TARGET fti_f90.static fti_f90.shared PROPERTY LINK_FLAGS " ${MPI_Fortran_LINK_FLAGS} ${MPI_C_LINK_FLAGS} ")
	set_property(TARGET fti_f90.static fti_f90.shared PROPERTY OUTPUT_NAME fti_f90)
	set_property(TARGET fti_f90.static PROPERTY Fortran_MODULE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include")

	install(TARGETS fti_f90.static fti_f90.shared DESTINATION lib EXPORT FTILib)
	install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/" DESTINATION include FILES_MATCHING PATTERN "*.mod")
endif()


# in tree config

set(FTI_INCLUDE_PATH "${CMAKE_CURRENT_BINARY_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE STRING "FTI Include path")
export(EXPORT FTILib FILE "${PROJECT_BINARY_DIR}/FTILib.cmake")
configure_file(CMakeScripts/FTIConfig.cmake.in "${PROJECT_BINARY_DIR}/FTIConfig.cmake" @ONLY)
export(PACKAGE FTI)


# installable config

set(WORK_DIR "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}")
set(FTI_INCLUDE_PATH "${CMAKE_INSTALL_PREFIX}/include")
install(EXPORT FTILib DESTINATION lib/CMake/FTI)
configure_file(CMakeScripts/FTIConfig.cmake.in "${WORK_DIR}/FTIConfig.cmake" @ONLY)
export(PACKAGE FTI)
install(FILES "${WORK_DIR}/FTIConfig.cmake" DESTINATION lib/CMake/FTI)
