cmake_minimum_required(VERSION 3.3)

include("${CMAKE_CURRENT_LIST_DIR}/FTILib.cmake")

if(NOT TARGET fti.shared)
	set(FTI_FOUND "FALSE")
	if(NOT "${FTI_FIND_QUIETLY}")
		message(WARNING "FTI: not found")
	endif()
endif()

# no need for that. This is now included in the library
set(FTI_INCLUDE_PATH)
