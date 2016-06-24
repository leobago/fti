##############################################################################
# Copyright (c) 2013-2014, Julien Bigot - CEA (julien.bigot@cea.fr)
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
##############################################################################

cmake_minimum_required(VERSION 2.8)
# protect the project call to workaround a cmake bug
if("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
	project(Bpp Fortran)
endif()

include(cmake/Bpp.cmake)

file(READ VERSION BPP_VERSION)
string(REGEX REPLACE "\n.*" "" BPP_VERSION "${BPP_VERSION}")

# in tree config

set(BPP_EXE "${CMAKE_CURRENT_BINARY_DIR}/bpp" CACHE PATH "BPP Executable")
set(BPP_CMAKE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Bpp.cmake")
set(BPP_DEFAULT_INCLUDES -I ${CMAKE_CURRENT_SOURCE_DIR}/include/ CACHE STRING "BPP default include path")

configure_file(cmake/BppConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/BppConfig.cmake" @ONLY)
configure_file(cmake/BppConfigVersion.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/BppConfigVersion.cmake" @ONLY)
configure_file(cmake/Bpp.cmake "${CMAKE_CURRENT_BINARY_DIR}/Bpp.cmake" COPYONLY)
configure_file(cmake/TestFortType.cmake "${CMAKE_CURRENT_BINARY_DIR}/TestFortType.cmake" COPYONLY)
configure_file(scripts/bpp.mk.in "${CMAKE_CURRENT_BINARY_DIR}/bpp.mk" @ONLY)
configure_file(scripts/bpp "${CMAKE_CURRENT_BINARY_DIR}/bpp" @ONLY)
export(PACKAGE Bpp)


# installable config

set(WORK_DIR "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}")

set(BPP_EXE "${CMAKE_INSTALL_PREFIX}/bin/bpp")
set(BPP_CMAKE "${CMAKE_INSTALL_PREFIX}lib/CMake/Bpp/Bpp.cmake")
set(BPP_DEFAULT_INCLUDES "-I ${CMAKE_INSTALL_PREFIX}/share/Bpp/include/")

configure_file(cmake/BppConfig.cmake.in "${WORK_DIR}/BppConfig.cmake" @ONLY)
configure_file(scripts/bpp.mk.in "${WORK_DIR}/bpp.mk" @ONLY)
install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/bpp
	DESTINATION bin
)
install(DIRECTORY include/
	DESTINATION share/Bpp/include/
	FILES_MATCHING PATTERN "*.bpp.sh"
)
install(FILES "${WORK_DIR}/bpp.mk"
	DESTINATION share/Bpp/
)
install(FILES
		cmake/Bpp.cmake
		cmake/TestFortType.cmake
		"${WORK_DIR}/BppConfig.cmake"
		"${CMAKE_CURRENT_BINARY_DIR}/BppConfigVersion.cmake"
	DESTINATION lib/CMake/Bpp
)
