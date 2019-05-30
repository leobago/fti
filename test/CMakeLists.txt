add_executable(addInArray addInArray.c)
target_link_libraries(addInArray fti.static)

add_executable(diffSizes diffSizes.c)
target_link_libraries(diffSizes fti.static)

if(ENABLE_HDF5)
	add_executable(hdf5Test hdf5Test.c)
	target_link_libraries(hdf5Test fti.static)

	add_executable(hdf5noFTI hdf5noFTI.c)
    target_link_libraries(hdf5noFTI ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})

    add_executable(hdf5CreateBasePattern hdf5CreateBasePattern.c)
    target_link_libraries(hdf5CreateBasePattern ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})
    file(COPY hdf5Test.sh DESTINATION .)
endif()

add_executable(heatdis heatdis.c)
target_link_libraries(heatdis fti.static)

add_executable(nodeFlag nodeFlag.c)
target_link_libraries(nodeFlag fti.static)

add_executable(corrupt corrupt.c)
target_link_libraries(corrupt fti.static)

add_executable(syncIntv syncIntv.c)
target_link_libraries(syncIntv fti.static)

add_subdirectory(local)
  
add_subdirectory(cornerCases)
add_subdirectory(ckptHierarchy)

if (ENABLE_GPU)
  message(AUTHOR_WARNING "  ** Project is going to be build WITH GPU Support")
  add_subdirectory(cuda)
endif()

file(COPY configs DESTINATION .)
file(COPY patterns DESTINATION .)
file(COPY README.txt DESTINATION .)
file(COPY tests.sh DESTINATION .)
file(COPY syncIntvTest.sh DESTINATION .)

