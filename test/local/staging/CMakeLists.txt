enable_testing()

list(APPEND test_labels_current "GIO")

# Install ITF Test Fixtures/Suites
foreach(tcase "staging")
    InstallSuiteITFConvention(${tcase} ${test_labels_current})
endforeach(tcase)

add_executable(massive.exe massive.c)
target_link_libraries(massive.exe fti.static ${MPI_C_LIBRARIES} m)
set_property(TARGET massive.exe APPEND PROPERTY COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
set_property(TARGET massive.exe APPEND PROPERTY LINK_FLAGS ${MPI_C_LINK_FLAGS})

list(REMOVE_ITEM test_labels_current "GIO")
