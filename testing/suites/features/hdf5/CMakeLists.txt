add_executable("hdf5noFTI.exe" "hdf5noFTI.c")
target_link_libraries("hdf5noFTI.exe" ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})
target_include_directories("hdf5noFTI.exe" PRIVATE ${HDF5_INCLUDE_DIRS})

add_executable("hdf5CreateBasePattern.exe" "hdf5CreateBasePattern.c")
target_link_libraries("hdf5CreateBasePattern.exe"
                    ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})
target_include_directories("hdf5CreateBasePattern.exe" PRIVATE
                    ${HDF5_INCLUDE_DIRS})

InstallTestApplication("hdf5Test.exe" "hdf5Test.c")
DeclareITFSuite("hdf5.itf" ${test_labels_current} "hdf5")
