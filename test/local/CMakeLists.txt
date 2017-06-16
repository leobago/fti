add_executable(check.exe check.c)
target_link_libraries(check.exe fti.static ${MPI_C_LIBRARIES} m)
set_property(TARGET check.exe APPEND PROPERTY COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
set_property(TARGET check.exe APPEND PROPERTY LINK_FLAGS ${MPI_C_LINK_FLAGS})

file(COPY Global Local Meta chk DESTINATION .)
install(FILES run-checks.sh PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ DESTINATION ../build/test/local)
