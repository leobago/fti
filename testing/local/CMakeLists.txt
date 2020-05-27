enable_testing()

if (ENABLE_GPU)
  add_subdirectory(dCPGPU)
  if(ENABLE_HDF5)
    add_subdirectory(hdf5GPU)
  endif()
endif()   


# ------------------------------ ITF Test Suites ------------------------------

list(APPEND test_labels_current "local")

add_subdirectory(recoverVar)
add_subdirectory(recoverName)
add_subdirectory(diffckpt)
add_subdirectory(standard)
add_subdirectory(staging)
add_subdirectory(keepL4Ckpt)

if(ENABLE_HDF5)
  add_subdirectory(variateProcessorRestart)
endif()

list(REMOVE_ITEM test_labels_current "local")

# -----------------------------------------------------------------------------

set(FULLPATH "${CMAKE_CURRENT_BINARY_DIR}")

if (ENABLE_GPU)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DGPUSUPPORT")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -DGPUSUPPORT")
  add_executable(check-gpu.exe check.cu)
  add_executable(checkAll.exe checkAll.c cudaTest.cu)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGPUSUPPORT")
endif()

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
    set( DCP_CHECK_DEF "-DENABLE_OPENSSL" )
endif()

configure_file(run-checks-f90.in ${CMAKE_CURRENT_SOURCE_DIR}/run-checks-f90.sh @ONLY)
configure_file(run-travis-locally.in ${CMAKE_CURRENT_SOURCE_DIR}/run-travis-locally.sh @ONLY)

# TODO: Deprecated when CUDA and Fortrain tests are merged in ITF
install(DIRECTORY DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Local)
install(DIRECTORY DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Global)
install(DIRECTORY DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Meta)
install(DIRECTORY DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/chk)

file(INSTALL run-checks-f90.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_WRITE)
file(INSTALL run-travis-locally.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_WRITE)

if (ENABLE_GPU)
  file(INSTALL run-cuda-checks.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_WRITE)
  file(INSTALL run-checks-all.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR} PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_WRITE)
  file(REMOVE run-cuda-checks.sh)
  file(REMOVE run-checks-all.sh)
endif()  

file(REMOVE run-checks-f90.sh)
file(REMOVE run-travis-locally.sh)
file(REMOVE check_functions.mod)
