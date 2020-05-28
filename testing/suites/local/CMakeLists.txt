list(APPEND test_labels_current "local")

add_subdirectory(recoverVar)
add_subdirectory(recoverName)
add_subdirectory(diffckpt)
add_subdirectory(standard)
add_subdirectory(staging)
add_subdirectory(cornerCases)
add_subdirectory(keepL4Ckpt)

if(ENABLE_HDF5)
  add_subdirectory(variateProcessorRestart)
endif()

list(REMOVE_ITEM test_labels_current "local")
