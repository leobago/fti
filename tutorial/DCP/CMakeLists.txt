add_executable(hdDCP.exe heatdis.c)
target_link_libraries(hdDCP.exe fti.static ${MPI_C_LIBRARIES} m)
target_include_directories(hdDCP.exe PUBLIC ${CMAKE_SOURCE_DIR}/include)
set_property(TARGET hdDCP.exe APPEND PROPERTY COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
set_property(TARGET hdDCP.exe APPEND PROPERTY LINK_FLAGS ${MPI_C_LINK_FLAGS})

set(L1_FREQ 1)
set(L2_FREQ 0)
set(L3_FREQ 0)
set(L4_FREQ 0)
set(LOCAL ./local)
set(GLOBAL ./global)
set(META ./meta)

configure_file(config.fti.in config.DCP.fti)

install(DIRECTORY DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/local)
install(DIRECTORY DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/global)
install(DIRECTORY DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/meta)

add_custom_target(hdDCP mpirun -n 8 --oversubscribe ./hdDCP.exe 64 ./config.DCP.fti)
add_custom_target(hdDCPrm rm -Rf ./local/* ./global/* ./meta/* COMMAND sed -i 's/failure[ ]\\+= 1/failure = 0/g' config.DCP.fti)
