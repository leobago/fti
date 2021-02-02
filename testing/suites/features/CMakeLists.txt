list(APPEND test_labels_current "features")

add_subdirectory(differentialCkpt)
add_subdirectory(recoverName)
add_subdirectory(recoverVar)
add_subdirectory(staging)
add_subdirectory(getConfig)
add_subdirectory(syncIntv)
add_subdirectory(useCases)
add_subdirectory(cornerCases)
add_subdirectory(keepL4Ckpt)
add_subdirectory(cppOperability)

if(ENABLE_HDF5)
  add_subdirectory(variateProcessorRestart)
  add_subdirectory(hdf5)
endif()

if(ENABLE_FORTRAN)
  add_subdirectory(fortran)
endif()

list(REMOVE_ITEM test_labels_current "features")

