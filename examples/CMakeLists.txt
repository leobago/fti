link_directories(${CMAKE_BINARY_DIR}/lib)

add_executable(hd.exe heatdis.c)
target_link_libraries(hd.exe ${CMAKE_BINARY_DIR}/vendor/fti/lib/libfti.a ${MPI_C_LIBRARIES} m)

if(MPI_C_COMPILE_FLAGS)
    set_target_properties(hd.exe PROPERTIES
        COMPILE_FLAGS "${MPI_C_COMPILE_FLAGS}")
endif()

if(MPI_C_LINK_FLAGS)
    set_target_properties(hd.exe PROPERTIES
        LINK_FLAGS "${MPI_C_LINK_FLAGS}")
endif()

add_executable(hd2.exe heatd2.c)
target_link_libraries(hd2.exe ${CMAKE_BINARY_DIR}/vendor/fti/lib/libfti.a ${MPI_C_LIBRARIES} m)

if(MPI_C_COMPILE_FLAGS)
    set_target_properties(hd2.exe PROPERTIES
        COMPILE_FLAGS "${MPI_C_COMPILE_FLAGS}")
endif()

if(MPI_C_LINK_FLAGS)
    set_target_properties(hd2.exe PROPERTIES
        LINK_FLAGS "${MPI_C_LINK_FLAGS}")
endif()

if(ENABLE_FORTRAN)
    add_executable(hdf.exe fheatdis.f90)
    target_link_libraries(hdf.exe fti_f90.shared ${CMAKE_BINARY_DIR}/vendor/fti/lib/libfti.a ${MPI_Fortran_LIBRARIES} m)

    if(MPI_Fortran_COMPILE_FLAGS)
        set_target_properties(hdf.exe PROPERTIES
            COMPILE_FLAGS "${MPI_Fortran_COMPILE_FLAGS}")
    endif()

    if(MPI_Fortran_LINK_FLAGS)
        set_target_properties(hdf.exe PROPERTIES
            LINK_FLAGS "${MPI_Fortran_LINK_FLAGS}")
    endif()
endif()
