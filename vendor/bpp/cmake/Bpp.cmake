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

include(${CMAKE_CURRENT_LIST_DIR}/TestFortType.cmake)

# A function to generate the BPP config.bpp.sh file
function(bpp_gen_config OUTFILE)
	foreach(TYPENAME "CHARACTER" "COMPLEX" "INTEGER" "LOGICAL" "REAL")
		foreach(TYPESIZE 1 2 4 8 16 32 64)
			test_fort_type("BPP_${TYPENAME}${TYPESIZE}_WORKS" "${TYPENAME}" "${TYPESIZE}")
			if ( "BPP_${TYPENAME}${TYPESIZE}_WORKS" )
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
function(bpp_preprocess OUTVAR FIRST_SRC)
	set(BPP_INCLUDE_PARAMS ${BPP_DEFAULT_INCLUDES})

	get_property(INCLUDE_DIRS DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
	foreach(INCLUDE_DIR ${INCLUDE_DIRS})
		set(BPP_INCLUDE_PARAMS ${BPP_INCLUDE_PARAMS} "-I" "${INCLUDE_DIR}")
	endforeach()

	bpp_gen_config("${CMAKE_CURRENT_BINARY_DIR}/bppconf/config.bpp.sh")
	set(BPP_INCLUDE_PARAMS ${BPP_INCLUDE_PARAMS} "-I" "${CMAKE_CURRENT_BINARY_DIR}/bppconf")


	set(OUTFILES)
	foreach(SRC "${FIRST_SRC}" ${ARGN})
		get_filename_component(OUTFILE "${SRC}" NAME)
		string(REGEX REPLACE "\\.[bB][pP][pP]$" "" OUTFILE "${OUTFILE}")
		set(OUTFILE "${CMAKE_CURRENT_BINARY_DIR}/${OUTFILE}")
		add_custom_command(OUTPUT "${OUTFILE}"
			COMMAND "${BPP_EXE}" ${BPP_INCLUDE_PARAMS} "${SRC}" "${OUTFILE}"
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
			MAIN_DEPENDENCY "${SRC}"
			VERBATIM
		)
		list(APPEND OUTFILES "${OUTFILE}")
	endforeach()

	set(${OUTVAR} "${OUTFILES}" PARENT_SCOPE)
endfunction()
