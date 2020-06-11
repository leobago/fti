list(APPEND test_labels_current "features")

add_subdirectory(differentialCkpt)
add_subdirectory(recoverName)
add_subdirectory(recoverVar)
add_subdirectory(staging)
add_subdirectory(getConfig)

if(ENABLE_HDF5)
    add_subdirectory(variateProcessorRestart)
    add_subdirectory(hdf5)
endif()

list(REMOVE_ITEM test_labels_current "features")
