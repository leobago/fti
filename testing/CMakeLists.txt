enable_testing()

# ------------------------ Install-Time Test Variables -----------------------

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

set(testing_dir ${CMAKE_CURRENT_BINARY_DIR})
add_subdirectory(tools/itf)

# Declares an ITF CTest and assign it labels with the remaining arguments
function(DeclareITFSuite FixtureName)

  message("ITF CTest configured: ${FixtureName} (${ARGN})")

  # Define the test rule using the fixtures
  add_test(NAME ${FixtureName} COMMAND
    ${itf_run_cmd} ${CMAKE_CURRENT_BINARY_DIR}/${FixtureName})

  # Attribute the a label to every test in the suite
  set_property(TEST ${FixtureName} PROPERTY LABELS ${ARGN})
  install(FILES ${FixtureName} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endfunction()

# Install a Test application with its source dependencies as extra arguments
function(InstallTestApplication AppName)
  add_executable(${AppName} ${ARGN})
  target_link_libraries(${AppName} fti.static ${MPI_C_LIBRARIES} m)
  set_property(TARGET ${AppName} APPEND PROPERTY COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
  set_property(TARGET ${AppName} APPEND PROPERTY LINK_FLAGS ${MPI_C_LINK_FLAGS})
endfunction()

# -------------------- Inclusion of ITF Test subdirectories -------------------

add_subdirectory(suites/core)
add_subdirectory(suites/features)

if(ENABLE_EXAMPLES)
  add_subdirectory(suites/examples)
endif()

