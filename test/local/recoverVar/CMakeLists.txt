enable_testing()

list(APPEND test_labels_current "recovervar")

# Install ITF Test Fixtures/Suites
foreach(tcase "recovervar")
    InstallSuiteITFConvention(${tcase} ${test_labels_current})
endforeach(tcase)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

add_executable(recoverVar.exe checkRecoverVar.c)
target_link_libraries(recoverVar.exe fti.static ${MPI_C_LIBRARIES} m)

set_property(TARGET recoverVar.exe PROPERTY C_STANDARD 99)  
set_property(TARGET recoverVar.exe APPEND PROPERTY COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
set_property(TARGET recoverVar.exe APPEND PROPERTY LINK_FLAGS ${MPI_C_LINK_FLAGS})

list(REMOVE_ITEM test_labels_current "recovervar")
