add_subdirectory(L1)
add_subdirectory(L2)
add_subdirectory(L3)
add_subdirectory(L4)
add_subdirectory(DCP)
add_subdirectory(practice)
add_subdirectory(experiment)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR} DESTINATION ${CMAKE_INSTALL_BINDIR}
				REGEX "CMakeFiles" EXCLUDE
				REGEX "cmake_install.cmake" EXCLUDE
				REGEX "Makefile" EXCLUDE)

