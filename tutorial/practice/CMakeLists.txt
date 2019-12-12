add_executable(hdp.exe heatdis.c)
target_link_libraries(hdp.exe fti.static ${MPI_C_LIBRARIES} m)
set_property(TARGET hdp.exe APPEND PROPERTY COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
set_property(TARGET hdp.exe APPEND PROPERTY LINK_FLAGS ${MPI_C_LINK_FLAGS})

configure_file(config.fti.in config.fti)
