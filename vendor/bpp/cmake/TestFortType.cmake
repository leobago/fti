################################################################################
# Copyright (c) 2013-2014, Julien Bigot - CEA (julien.bigot@cea.fr)
# All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
################################################################################

function(test_fort_type RESULT_VAR TYPE KIND)
	if(DEFINED "${RESULT_VAR}")
		return()
	endif()
	message(STATUS "Checking whether Fortran supports type ${TYPE}(KIND=${KIND})")
	set(TEST_FILE "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/cmake_test_${RESULT_VAR}.f90")
	file(WRITE "${TEST_FILE}" "
program test_${RESULT_VAR}
  ${TYPE}(KIND=${KIND}):: tstvar
end program test_${RESULT_VAR}
")
	try_compile(COMPILE_RESULT "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}" "${TEST_FILE}"
		OUTPUT_VARIABLE COMPILE_OUTPUT
	)
	if(COMPILE_RESULT)
		file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
		"${TYPE}(KIND=${KIND}) type successfully compiled with the following output:\n"
		"${COMPILE_OUTPUT}\n")
		message(STATUS "Checking whether Fortran supports type ${TYPE}(KIND=${KIND}) -- yes")
	else()
		file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
		"${TYPE}(KIND=${KIND}) type failed to compile with the following output:\n"
		"${COMPILE_OUTPUT}\n")
		message(STATUS "Checking whether Fortran supports type ${TYPE}(KIND=${KIND}) -- no")
	endif()
	set("${RESULT_VAR}" "${COMPILE_RESULT}" CACHE BOOL "Whether Fortran supports type ${TYPE}(KIND=${KIND})")
	mark_as_advanced("${RESULT_VAR}")
endfunction()

function(test_fort_hdf5_type RESULT_VAR TYPE KIND)
	if(DEFINED "${RESULT_VAR}")
		return()
	endif()
	find_package(HDF5)
	if(NOT "${HDF5_FOUND}")
		set("${RESULT_VAR}" "HDF5-NOTFOUND" CACHE STRING "HDF5 constant for Fortran type ${TYPE}(KIND=${KIND})")
	mark_as_advanced("${RESULT_VAR}")
		return()
	endif()
	message(STATUS "Detecting HDF5 constant for Fortran type ${TYPE}(KIND=${KIND})")
	set(TEST_FILE "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/cmake_test_${RESULT_VAR}.f90")
	if("${TYPE}" STREQUAL "INTEGER")
		math(EXPR SIZE "8*${KIND}")
		set(H5CST "H5T_STD_I${SIZE}LE")
	elseif("${TYPE}" STREQUAL "REAL")
		math(EXPR SIZE "8*${KIND}")
		set(H5CST "H5T_IEEE_F${SIZE}LE")
	else()
		set(H5CST "HDF5_CONSTANT-NOTFOUND")
	endif()
	file(WRITE "${TEST_FILE}" "
program test_${RESULT_VAR}
  use hdf5
  ${TYPE}(KIND=${KIND}):: tstvar
  integer(HID_T):: h5var
  h5var = ${H5CST}
end program test_${RESULT_VAR}
")
	try_compile(COMPILE_RESULT "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}" "${TEST_FILE}"
		CMAKE_FLAGS
			"-DINCLUDE_DIRECTORIES=${HDF5_Fortran_INCLUDE_PATH}"
			"-DCMAKE_Fortran_FLAGS=${HDF5_Fortran_COMPILE_FLAGS}"
			"-DCMAKE_EXE_LINKER_FLAGS=${HDF5_Fortran_LINK_FLAGS}"
		LINK_LIBRARIES
			${HDF5_Fortran_LIBRARIES} ${HDF5_C_LIBRARIES}
		OUTPUT_VARIABLE COMPILE_OUTPUT
	)
	if(COMPILE_RESULT)
		file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeOutput.log
		"Fortran/HDF5 ${TYPE}(KIND=${KIND}) type successfully compiled with the following output:\n"
		"${COMPILE_OUTPUT}\n")
		set("${RESULT_VAR}" "${H5CST}" CACHE STRING "HDF5 constant for Fortran type ${TYPE}(KIND=${KIND})")
	else()
		file(APPEND ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log
		"Fortran/HDF5 ${TYPE}(KIND=${KIND}) type failed to compile with the following output:\n"
		"${COMPILE_OUTPUT}\n")
		set("${RESULT_VAR}" "HDF5_CONSTANT-NOTFOUND" CACHE STRING "HDF5 constant for Fortran type ${TYPE}(KIND=${KIND})")
	endif()
	message(STATUS "Detecting HDF5 constant for Fortran type ${TYPE}(KIND=${KIND}) -- ${${RESULT_VAR}}")
	mark_as_advanced("${RESULT_VAR}")
endfunction()
