add_executable(hdex.exe heatdis.c)
target_link_libraries(hdex.exe fti.static ${MPI_C_LIBRARIES} m)
set_property(TARGET hdex.exe APPEND PROPERTY COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
set_property(TARGET hdex.exe APPEND PROPERTY LINK_FLAGS ${MPI_C_LINK_FLAGS})

set(LOCAL ./local)
set(GLOBAL ./global)
set(META ./meta)

configure_file(config.fti.in config.fti)

install(DIRECTORY DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/local)
install(DIRECTORY DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/global)
install(DIRECTORY DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/meta)
