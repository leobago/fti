add_executable(checkDCPPosix.exe checkDCPPosix.c)
set(CMAKE_VERBOSE_MAKEFILE on)
target_link_libraries(checkDCPPosix.exe fti.static ${MPI_C_LIBRARIES} m)
set_property(TARGET checkDCPPosix.exe APPEND PROPERTY COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
set_property(TARGET checkDCPPosix.exe APPEND PROPERTY LINK_FLAGS ${MPI_C_LINK_FLAGS})
configure_file(run.sh.in ${CMAKE_CURRENT_SOURCE_DIR}/run.sh @ONLY)

install(DIRECTORY DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Local)
install(DIRECTORY DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Global)
install(DIRECTORY DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Meta)
install(DIRECTORY DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/chk)

file(COPY cfg DESTINATION .)
file(INSTALL run.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_WRITE)
file(REMOVE run-checkDCPPosixs.sh)
  
# This will install the diff_test application directly without the need for a makefile
add_executable(diff_test.exe diff_test_func.c diff_test.c diff_test.h)
target_link_libraries(diff_test.exe fti.static ${MPI_C_LIBRARIES} m)
set_property(TARGET diff_test.exe APPEND PROPERTY COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
set_property(TARGET diff_test.exe APPEND PROPERTY LINK_FLAGS ${MPI_C_LINK_FLAGS})