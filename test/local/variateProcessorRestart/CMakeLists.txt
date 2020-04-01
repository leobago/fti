enable_testing()

list(APPEND test_labels_current "vpr")

# Install ITF Test Fixtures/Suites
foreach(tcase "vpr")
    InstallSuiteITFConvention(${tcase} ${test_labels_current})
endforeach(tcase)

add_executable(test.exe test.c)
target_link_libraries(test.exe fti.static ${MPI_C_LIBRARIES} m)
set_property(TARGET test.exe APPEND PROPERTY COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
set_property(TARGET test.exe APPEND PROPERTY LINK_FLAGS ${MPI_C_LINK_FLAGS})

list(REMOVE_ITEM test_labels_current "vpr")