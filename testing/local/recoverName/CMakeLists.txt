enable_testing()

list(APPEND test_labels_current "recovername")

# Install ITF Test Fixtures/Suites
foreach(tcase "recovername")
    InstallSuiteITFConvention(${tcase} ${test_labels_current})
endforeach(tcase)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
add_executable(recoverName.exe checkRecoverName.c)


target_link_libraries(recoverName.exe fti.static ${MPI_C_LIBRARIES} m)

set_property(TARGET recoverName.exe PROPERTY C_STANDARD 99)  
set_property(TARGET recoverName.exe APPEND PROPERTY COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
set_property(TARGET recoverName.exe APPEND PROPERTY LINK_FLAGS ${MPI_C_LINK_FLAGS})
#set_property(TARGET heatdis.exe APPEND PROPERTY COMPILE_FLAGS "-L${CMAKE_INSTALL_PREFIX}/include")

list(REMOVE_ITEM test_labels_current "recovername")
