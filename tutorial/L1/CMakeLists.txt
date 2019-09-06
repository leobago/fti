add_executable(hdl1.exe heatdis.c)
target_link_libraries(hdl1.exe fti.static ${MPI_C_LIBRARIES} m)
set_property(TARGET hdl1.exe APPEND PROPERTY COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
set_property(TARGET hdl1.exe APPEND PROPERTY LINK_FLAGS ${MPI_C_LINK_FLAGS})

set(L1_FREQ 1)
set(L2_FREQ 0)
set(L3_FREQ 0)
set(L4_FREQ 0)
set(LOCAL ./local)
set(GLOBAL ./global)
set(META ./meta)

configure_file(config.fti.in config.L1.fti)

install(DIRECTORY DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/local)
install(DIRECTORY DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/global)
install(DIRECTORY DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/meta)

add_custom_target(hdl1 mpirun -n 8 ./hdl1.exe 64 ./config.L1.fti)
add_custom_target(hdl1rm rm -Rf ./local/* ./global/* ./meta/* COMMAND sed -i 's/failure[ ]\\+= 1/failure = 0/g' config.L1.fti)
