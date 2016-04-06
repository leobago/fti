cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
	project(FTI C Fortran)
endif()

option(ENABLE_FORTRAN "Enables the generation of the Fortran wrapper for FTI" OFF)
option(ENABLE_EXAMPLES "Enables the generation of examples" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeScripts")
include(AppendProperty)
include(BPP)
include(FortranCInterface)

find_package(MPI REQUIRED)

add_subdirectory(deps)

include_directories("${CMAKE_CURRENT_BINARY_DIR}"
	"${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}/include"
	${MPI_Fortran_INCLUDE_PATH} ${MPI_C_INCLUDE_PATH})

set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include)

set(SRC_FTI
	src/api.c src/checkpoint.c src/conf.c src/meta.c
	src/postckpt.c src/postreco.c src/recover.c
	src/tools.c src/topo.c)

append_property(SOURCE ${SRC_FTI}
	PROPERTY COMPILE_FLAGS "${MPI_C_COMPILE_FLAGS}")

add_library(fti.static STATIC ${SRC_FTI}
    $<TARGET_OBJECTS:iniparser> $<TARGET_OBJECTS:jerasure>)
add_library(fti.shared SHARED ${SRC_FTI}
    $<TARGET_OBJECTS:iniparser> $<TARGET_OBJECTS:jerasure>)

append_property(TARGET fti.static fti.shared
	PROPERTY LINK_FLAGS "${MPI_C_LINK_FLAGS}")
set_property(TARGET fti.static fti.shared
	PROPERTY OUTPUT_NAME fti)

target_link_libraries(fti.static ${MPI_C_LIBRARIES})
target_link_libraries(fti.shared ${MPI_C_LIBRARIES})

install(TARGETS fti.static fti.shared
	DESTINATION lib)
install(FILES "include/fti.h"
	DESTINATION include)

if(ENABLE_FORTRAN)
	bpp_preprocess(BPP_FTI_F90
		src/fortran/interface.F90.bpp)
	set(SRC_FTI_F90 ${BPP_FTI_F90}
		src/fortran/ftif.c)
	append_property(SOURCE ${SRC_FTI_F90}
		PROPERTY COMPILE_FLAGS "${MPI_Fortran_COMPILE_FLAGS}")

	add_library(fti_f90.static STATIC ${SRC_FTI_F90})
	target_link_libraries(fti_f90.static
		fti.static ${MPI_Fortran_LIBRARIES} ${MPI_C_LIBRARIES})

	add_library(fti_f90.shared SHARED ${SRC_FTI_F90})
	target_link_libraries(fti_f90.shared
		fti.shared ${MPI_Fortran_LIBRARIES} ${MPI_C_LIBRARIES})

	append_property(TARGET fti_f90.static fti_f90.shared
		PROPERTY LINK_FLAGS "${MPI_Fortran_LINK_FLAGS} ${MPI_C_LINK_FLAGS}")
	set_property(TARGET fti_f90.static fti_f90.shared
		PROPERTY OUTPUT_NAME fti_f90)

	install(TARGETS fti_f90.static fti_f90.shared
		DESTINATION lib)
	install(FILES ${CMAKE_Fortran_MODULE_DIRECTORY}/fti.mod
		DESTINATION include)
endif()

if(NOT "${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
	set(FTI_INCLUDE_PATH "${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/include" PARENT_SCOPE)
endif()

if(ENABLE_EXAMPLES)
	add_subdirectory(examples)
endif()
