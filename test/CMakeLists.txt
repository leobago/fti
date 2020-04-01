
add_executable(addInArray addInArray.c)
target_link_libraries(addInArray fti.static)

add_executable(diffSizes diffSizes.c)
target_link_libraries(diffSizes fti.static)

if(ENABLE_HDF5)
	add_executable(hdf5Test hdf5Test.c)
	target_link_libraries(hdf5Test fti.static)

	add_executable(hdf5noFTI hdf5noFTI.c)
    target_link_libraries(hdf5noFTI ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})

    add_executable(hdf5CreateBasePattern hdf5CreateBasePattern.c)
    target_link_libraries(hdf5CreateBasePattern ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})
    file(COPY hdf5Test.sh DESTINATION .)
endif()

add_executable(heatdis heatdis.c)
target_link_libraries(heatdis fti.static)

add_executable(nodeFlag nodeFlag.c)
target_link_libraries(nodeFlag fti.static)

add_executable(corrupt corrupt.c)
target_link_libraries(corrupt fti.static)

add_executable(syncIntv syncIntv.c)
target_link_libraries(syncIntv fti.static)
  
add_subdirectory(cornerCases)
add_subdirectory(ckptHierarchy)

if (ENABLE_GPU)
  add_subdirectory(cuda)
endif()

file(COPY configs DESTINATION .)
file(COPY patterns DESTINATION .)
file(COPY README.txt DESTINATION .)
file(COPY tests.sh DESTINATION .)
file(COPY syncIntvTest.sh DESTINATION .)

# --------------------------- ITF engine Integration --------------------------

# The test directory
set(TEST_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(itf)

function(DeclareITFSuite FixtureName)
  # Declares an ITF CTest and assign it label with the remaining parameters

  message("Adding CTests calling ITF for fixture: ${FixtureName}")

  # Define the test rule using the fixtures
  add_test(NAME ${FixtureName} COMMAND
    ${itf_run_cmd} ${CMAKE_CURRENT_BINARY_DIR}/${FixtureName})

  # Attribute the a label to every test in the suite
  set_property(TEST ${FixtureName} PROPERTY LABELS ${ARGN})

  message("CTest Suite ${FixtureName} has the following labels: ${FixtureName} ${ARGN}")
endfunction()

function(InstallSuiteITFConvention FixturePreffix)
  # Install a fixture and create the appropriate test suite for it
  
  configure_file("${FixturePreffix}.fixture" ${CMAKE_CURRENT_BINARY_DIR} @ONLY)
  configure_file("${FixturePreffix}.suite" ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
  DeclareITFSuite("${FixturePreffix}" ${ARGN})
endfunction()

# ---------------------- Inclusion of Test subdirectories ---------------------

set(test_subdirs "local")

foreach(test_dir ${test_subdirs})
  add_subdirectory(${test_dir})
endforeach(test_dir)
