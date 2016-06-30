# A function to preprocess a source file with BPP
function(bpp_preprocess OUTVAR FIRST_SRC)
	set(RESULT)
	foreach(SRC "${FIRST_SRC}" ${ARGN})
		get_filename_component(OUTFILE "${SRC}" NAME)
		string(REGEX REPLACE "\\.[bB][pP][pP]$" "" OUTFILE "${OUTFILE}")
		set(OUTFILE "${CMAKE_CURRENT_BINARY_DIR}/${OUTFILE}")
		list(APPEND RESULT "${OUTFILE}")
		add_custom_command(OUTPUT "${OUTFILE}"
			COMMAND "${CMAKE_SOURCE_DIR}/vendor/fti/scripts/bpp" ARGS "${SRC}" "${OUTFILE}"
			WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
			MAIN_DEPENDENCY "${SRC}"
			VERBATIM
		)
	endforeach()
	set(${OUTVAR} ${RESULT} PARENT_SCOPE)
endfunction()
