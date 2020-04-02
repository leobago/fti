enable_testing()

# Add the local tag to all CTests defined here
set(diff_test_labels "dCP" "iCP")
list(APPEND test_labels_current ${diff_test_labels})

# ITF Test Fixture names
set(itf_dcp_fixtures "dCP-standard" "dCP-corrupt")

# Install ITF Test Fixtures/Suites
foreach(tcase ${itf_dcp_fixtures})
    InstallSuiteITFConvention(${tcase} ${test_labels_current})
endforeach(tcase)

set(CMAKE_VERBOSE_MAKEFILE on)

add_executable(checkDCPPosix.exe checkDCPPosix.c)
target_link_libraries(checkDCPPosix.exe fti.static ${MPI_C_LIBRARIES} m)
set_property(TARGET checkDCPPosix.exe APPEND PROPERTY COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
set_property(TARGET checkDCPPosix.exe APPEND PROPERTY LINK_FLAGS ${MPI_C_LINK_FLAGS})

add_executable(diff_test.exe diff_test_func.c diff_test.c diff_test.h)
target_link_libraries(diff_test.exe fti.static ${MPI_C_LIBRARIES} m)
set_property(TARGET diff_test.exe APPEND PROPERTY COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
set_property(TARGET diff_test.exe APPEND PROPERTY LINK_FLAGS ${MPI_C_LINK_FLAGS})

list(REMOVE_ITEM test_labels_current ${diff_test_labels})
