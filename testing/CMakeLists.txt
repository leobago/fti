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

# ------------------------ Install-Time Test Variables -----------------------

# TODO: deprecated when CUDA and FORTRAN tests are integrated into ITF
set(IO 1)
set(IO_NAMES "POSIX MPIIO FTIFF")
set(IO_IDS "1 2 3")
set(MEMORYTYPES "CPU")

if(ENABLE_SIONLIB)
  MATH(EXPR IO "${IO}+1")
  set(IO_NAMES "${IO_NAMES} SIONLIB")
  set(IO_IDS "${IO_IDS} 4")
endif()

if (ENABLE_HDF5)
  MATH(EXPR IO "${IO}+1")
  set(IO_NAMES "${IO_NAMES} HDF5")
  set(IO_IDS "${IO_IDS} 5")
endif()

# --------------------------- ITF engine Integration --------------------------

set(TEST_DIR ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(itf)

# Declares an ITF CTest and assign it labels with the remaining arguments
function(DeclareITFSuite FixtureName)

  message("ITF CTest installed: ${FixtureName} (${ARGN})")

  # Define the test rule using the fixtures
  add_test(NAME ${FixtureName} COMMAND
    ${itf_run_cmd} ${CMAKE_CURRENT_BINARY_DIR}/${FixtureName})

  # Attribute the a label to every test in the suite
  set_property(TEST ${FixtureName} PROPERTY LABELS ${ARGN})
endfunction()

# Install a fixture and create the appropriate test suite for it
function(InstallSuiteITFConvention FixturePreffix)  
  configure_file("${FixturePreffix}.fixture" ${CMAKE_CURRENT_BINARY_DIR} @ONLY)
  configure_file("${FixturePreffix}.suite" ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
  DeclareITFSuite("${FixturePreffix}" ${ARGN})
endfunction()

# Install a Test application with its source dependencies as extra arguments
function(InstallTestApplication AppName)
  add_executable(${AppName} ${ARGN})
  target_link_libraries(${AppName} fti.static ${MPI_C_LIBRARIES} m)
  set_property(TARGET ${AppName} APPEND PROPERTY COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
  set_property(TARGET ${AppName} APPEND PROPERTY LINK_FLAGS ${MPI_C_LINK_FLAGS})
endfunction()

# ---------------------- Inclusion of Test subdirectories ---------------------

set(test_subdirs "local")

foreach(test_dir ${test_subdirs})
  add_subdirectory(${test_dir})
endforeach(test_dir)

# -------------------------- Installing Tests for CI --------------------------

configure_file(ci/localtests.sh ${CMAKE_CURRENT_BINARY_DIR}/ci/localtests.sh COPYONLY)