list(APPEND test_labels_current "core")

add_subdirectory(ckptDiffSizes)
add_subdirectory(multiLevelCkpt)

list(REMOVE_ITEM test_labels_current "core")
