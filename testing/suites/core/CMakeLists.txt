list(APPEND test_labels_current "core")

add_subdirectory(ckptDiffSizes)
add_subdirectory(cornerCases)
add_subdirectory(keepL4Ckpt)
add_subdirectory(multiLevelCkpt)
add_subdirectory(syncIntv)
add_subdirectory(useCases)

list(REMOVE_ITEM test_labels_current "core")
