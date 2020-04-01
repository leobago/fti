enable_testing()

# TODO: Deprecated after full ITF merge
set(IO 1)
set(IO_NAMES "POSIX MPIIO FTIFF")
set(IO_IDS "1 2 3")
set(MEMORYTYPES "CPU")

# TODO: Deprecated after full ITF merge
if(ENABLE_HDF5)
  SET(DO_VPR_CHECK true)
else()
  SET(DO_VPR_CHECK false)
endif()

# TODO: Deprecated after full ITF merge
if(ENABLE_SIONLIB)
  MATH(EXPR IO "${IO}+1")
  set(IO_NAMES "${IO_NAMES} SIONLIB")
  set(IO_IDS "${IO_IDS} 4")
endif()

# TODO: Deprecated after full ITF merge
if (ENABLE_HDF5)
  MATH(EXPR IO "${IO}+1")
  set(IO_NAMES "${IO_NAMES} HDF5")
  set(IO_IDS "${IO_IDS} 5")
endif()

if (ENABLE_GPU)
  add_subdirectory(dCPGPU)
  if(ENABLE_HDF5)
    add_subdirectory(hdf5GPU)
  endif()
endif()   


# ------------------------------ ITF Test Suites ------------------------------

list(APPEND test_labels_current "local")

set(test_local_suites
  "recoverVar" "recoverName" "diffckpt"
  "staging" "keepL4Ckpt")

# Include all local test suites
foreach(tcase ${test_local_suites})
  add_subdirectory(${tcase})
endforeach(tcase)

# Conditiona tests
if(ENABLE_HDF5)
  add_subdirectory("variateProcessorRestart")
endif()


# TODO: Refactor this when ITF is fully merged
list(APPEND test_labels_current "standard")
InstallSuiteITFConvention("standard" ${test_labels_current})
InstallSuiteITFConvention("standard-disrupt" ${test_labels_current})
list(REMOVE_ITEM test_labels_current "standard")

# -----------------------------------------------------------------------------

set(FULLPATH "${CMAKE_CURRENT_BINARY_DIR}")

if (ENABLE_GPU)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DGPUSUPPORT")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DGPUSUPPORT")
#  add_executable(check.exe checkAll.c cudaTest.cu)
  add_executable(check.exe check.c)
  add_executable(check-gpu.exe check.cu)
  add_executable(checkAll.exe checkAll.c cudaTest.cu)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGPUSUPPORT")
else ()
  add_executable(check.exe check.c)
endif()
set(CMAKE_VERBOSE_MAKEFILE on)

target_link_libraries(check.exe fti.static ${MPI_C_LIBRARIES} m)
set_property(TARGET check.exe APPEND PROPERTY COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
set_property(TARGET check.exe APPEND PROPERTY LINK_FLAGS ${MPI_C_LINK_FLAGS})

if (ENABLE_GPU)
  target_link_libraries(check-gpu.exe fti.static ${MPI_C_LIBRARIES} m)
  target_link_libraries(checkAll.exe fti.static ${MPI_C_LIBRARIES} m)
  set_property(TARGET check-gpu.exe APPEND PROPERTY COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
  set_property(TARGET checkAll.exe APPEND PROPERTY LINK_FLAGS ${MPI_C_LINK_FLAGS})
endif()
 

if(ENABLE_FORTRAN)
    add_executable(fcheck.exe check.f90)
    target_link_libraries(fcheck.exe fti_f90.static ${MPI_Fortran_LIBRARIES} m)
    set_property(TARGET fcheck.exe APPEND PROPERTY COMPILE_FLAGS ${MPI_Fortran_COMPILE_FLAGS})
    set_property(TARGET fcheck.exe APPEND PROPERTY LINK_FLAGS ${MPI_Fortran_LINK_FLAGS})
    if(NOT MPI_USE_MOD)
	    target_compile_definitions(fcheck.exe PRIVATE -DMPI_USE_HEADER)
    endif()
	target_compile_options(fcheck.exe PRIVATE -cpp)
endif()


set(DIR ${CMAKE_BINARY_DIR}/)

configure_file(run-checks.in ${CMAKE_CURRENT_SOURCE_DIR}/run-checks.sh @ONLY)

if (ENABLE_GPU)
  set(MEMORYTYPES "${MEMORYTYPES} GPU")
  set(MEMORYTYPES "${MEMORYTYPES} UNIFIED")
  configure_file(run-cuda-checks.in ${CMAKE_CURRENT_SOURCE_DIR}/run-cuda-checks.sh @ONLY)
  configure_file(run-checks-all.in ${CMAKE_CURRENT_SOURCE_DIR}/run-checks-all.sh @ONLY)
endif()  

if ( OPENSSL_FOUND ) 
    set( DCP_CHECK_MD5_LIB "-lcrypto" )
    set( DCP_CHECK_DEF "" )
else()
    set( DCP_CHECK_MD5_LIB "" )
    set( DCP_CHECK_DEF "-DNO_OPENSSL" )
endif()

list(APPEND test_current_labels )

configure_file(run-checks-f90.in ${CMAKE_CURRENT_SOURCE_DIR}/run-checks-f90.sh @ONLY)
configure_file(run-travis-locally.in ${CMAKE_CURRENT_SOURCE_DIR}/run-travis-locally.sh @ONLY)

install(DIRECTORY DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Local)
install(DIRECTORY DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Global)
install(DIRECTORY DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Meta)
install(DIRECTORY DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/chk)

file(COPY Global Local Meta chk runchecks.cfg DESTINATION .)

file(INSTALL run-checks.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_WRITE)
file(INSTALL run-checks-f90.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_WRITE)
file(INSTALL run-travis-locally.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_WRITE)

file(REMOVE run-checks.sh)
  
if (ENABLE_GPU)
  file(INSTALL run-cuda-checks.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_WRITE)
  file(INSTALL run-checks-all.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_WRITE)
  file(REMOVE run-cuda-checks.sh)
  file(REMOVE run-checks-all.sh)
endif()  

file(REMOVE run-checks-f90.sh)
file(REMOVE run-travis-locally.sh)
file(REMOVE check_functions.mod)

list(REMOVE_ITEM test_labels_current "local")