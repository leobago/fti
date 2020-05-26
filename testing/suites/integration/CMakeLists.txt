list(APPEND test_labels_current "integration")

add_subdirectory(diffSizes)
add_subdirectory(syncintv)
add_subdirectory(useCases)

if(ENABLE_HDF5)
    add_subdirectory(hdf5)
endif()

list(REMOVE_ITEM test_labels_current "integration")
