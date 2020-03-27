add_executable(testkeepl4.exe test.c)
target_link_libraries(testkeepl4.exe fti.static ${MPI_C_LIBRARIES} m)
set_property(TARGET testkeepl4.exe APPEND PROPERTY COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
set_property(TARGET testkeepl4.exe APPEND PROPERTY LINK_FLAGS ${MPI_C_LINK_FLAGS})