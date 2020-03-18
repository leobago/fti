################################################################################
# Copyright (c) 2013-2019, Julien Bigot - CEA (julien.bigot@cea.fr)
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

cmake_minimum_required(VERSION 2.8)
cmake_policy(PUSH)


# Compute the installation prefix relative to this file.
get_filename_component(_BPP_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(_BPP_IMPORT_PREFIX "${_BPP_IMPORT_PREFIX}" PATH)
get_filename_component(_BPP_IMPORT_PREFIX "${_BPP_IMPORT_PREFIX}" PATH)
get_filename_component(_BPP_IMPORT_PREFIX "${_BPP_IMPORT_PREFIX}" PATH)
if(_BPP_IMPORT_PREFIX STREQUAL "/")
	set(_BPP_IMPORT_PREFIX "")
endif()


# A function to generate the BPP config.bpp.sh file
function(bpp_gen_config OUTFILE)
	include("${_BPP_IMPORT_PREFIX}/share/bpp/cmake/TestFortType.cmake")
	foreach(TYPENAME "CHARACTER" "COMPLEX" "INTEGER" "LOGICAL" "REAL")
		foreach(TYPESIZE 1 2 4 8 16 32 64)
			test_fort_type("BPP_${TYPENAME}${TYPESIZE}_WORKS" "${TYPENAME}" "${TYPESIZE}")
			if("${BPP_${TYPENAME}${TYPESIZE}_WORKS}")
				set(BPP_FORTTYPES "${BPP_FORTTYPES}${TYPENAME}${TYPESIZE} ")
			endif()
		endforeach()
	endforeach()
	if(NOT EXISTS "${OUTFILE}")
		file(WRITE "${OUTFILE}"
"# All types supported by the current Fortran implementation
BPP_FORTTYPES=\"${BPP_FORTTYPES}\"
# for compatibility
FORTTYPES=\"\${BPP_FORTTYPES}\"
")
	endif()
endfunction()



# A function to preprocess a source file with BPP
function(bpp_preprocess)
	cmake_parse_arguments(BPP_PREPROCESS "" "OUTPUT" "DEFINES;INCLUDES;SOURCES" "${FIRST_SRC}" ${ARGN})

	# old function signature for compatibility
	if ( 
			"${BPP_PREPROCESS_OUTPUT}" STREQUAL ""
			AND "${BPP_PREPROCESS_DEFINES}" STREQUAL ""
			AND "${BPP_PREPROCESS_INCLUDES}" STREQUAL ""
			AND "${BPP_PREPROCESS_SOURCES}" STREQUAL ""
	)
		list(GET ARGV 0 BPP_PREPROCESS_OUTPUT)
		list(REMOVE_AT ARGV 0)
		set(BPP_PREPROCESS_SOURCES ${ARGV})
	elseif(NOT "${BPP_PREPROCESS_UNPARSED_ARGUMENTS}" STREQUAL "")
		message(SEND_ERROR "Unexpected argument(s) to bpp_preprocess: ${BPP_PREPROCESS_UNPARSED_ARGUMENTS}")
	endif()
	
	unset(BPP_INCLUDE_PARAMS)

	get_property(DIR_INCLUDE_DIRS DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
	foreach(INCLUDE_DIR ${DIR_INCLUDE_DIRS} ${BPP_PREPROCESS_INCLUDES})
		set(BPP_INCLUDE_PARAMS ${BPP_INCLUDE_PARAMS} "-I" "${INCLUDE_DIR}")
	endforeach()
	foreach(DEFINE ${BPP_PREPROCESS_DEFINES})
		set(BPP_INCLUDE_PARAMS ${BPP_INCLUDE_PARAMS} "-D" "${DEFINE}")
	endforeach()

	bpp_gen_config("${CMAKE_CURRENT_BINARY_DIR}/bppconf/config.bpp.sh")
	set(BPP_INCLUDE_PARAMS ${BPP_INCLUDE_PARAMS} "-I" "${CMAKE_CURRENT_BINARY_DIR}/bppconf")
	
	set(OUTFILES)
	foreach(SRC ${BPP_PREPROCESS_SOURCES})
		get_filename_component(OUTFILE "${SRC}" NAME)
		string(REGEX REPLACE "\\.[bB][pP][pP]$" "" OUTFILE "${OUTFILE}")
		set(OUTFILE "${CMAKE_CURRENT_BINARY_DIR}/${OUTFILE}")
		add_custom_command(OUTPUT "${OUTFILE}"
			COMMAND "${_BPP_IMPORT_PREFIX}/bin/bpp" ${BPP_INCLUDE_PARAMS} "${SRC}" "${OUTFILE}"
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
			MAIN_DEPENDENCY "${SRC}"
			VERBATIM
		)
		list(APPEND OUTFILES "${OUTFILE}")
	endforeach()

	set(${BPP_PREPROCESS_OUTPUT} "${OUTFILES}" PARENT_SCOPE)
endfunction()

cmake_policy(POP)
