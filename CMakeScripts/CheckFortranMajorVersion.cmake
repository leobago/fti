cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

include(FortranCInterface)
include(CheckFortranCompilerFlag)

function(CheckCFortranMatch OUTPUT_VAR)
    if(NOT "${CMAKE_Fortran_COMPILER_VERSION}")

        message(AUTHOR_WARNING "
        ** Cmake variable 'CMAKE_Fortran_COMPILER_VERSION' is unset
        *  attempt to determine it manually...")

        if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "Intel")
            set(VER_CHECK_SRC "${CMAKE_CURRENT_LIST_DIR}/compiler_checks/intel_major_ver.f90")
        elseif("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU")
            set(VER_CHECK_SRC "${CMAKE_CURRENT_LIST_DIR}/compiler_checks/gnu_major_ver.f90")
        elseif("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "PGI")
            set(VER_CHECK_SRC "${CMAKE_CURRENT_LIST_DIR}/compiler_checks/pgi_major_ver.f90")
        endif()

        set(CMAKE_Fortran_FLAGS "-cpp")
        try_run( PROG_RAN COMPILE_SUCCESS
            "${CMAKE_BINARY_DIR}" "${VER_CHECK_SRC}"
            RUN_OUTPUT_VARIABLE VER_STRING
            )
        
        if ( "${PROG_RAN}" )
            string(REGEX MATCH "[0-9]+" CMAKE_Fortran_COMPILER_VERSION_MAJOR "${VER_STRING}")
            message(AUTHOR_WARNING "
            ** The major version was determined as: ${VER_STRING}")
        else()
            set(CMAKE_Fortran_COMPILER_VERSION_MAJOR "")
            message(AUTHOR_WARNING "
            ** The Fortran version could not be determined!")
        endif()
    
    else()

        string(REGEX MATCH "[0-9]+" CMAKE_Fortran_COMPILER_VERSION_MAJOR ${CMAKE_Fortran_COMPILER_VERSION} )

    endif()
    
    if("${CMAKE_Fortran_COMPILER_VERSION_MAJOR}")
        string(REGEX MATCH "[0-9]+" CMAKE_C_COMPILER_VERSION_MAJOR ${CMAKE_C_COMPILER_VERSION} )
        if("${CMAKE_C_COMPILER_ID}_${CMAKE_C_COMPILER_VERSION_MAJOR}" STREQUAL "${CMAKE_Fortran_COMPILER_ID}_${CMAKE_Fortran_COMPILER_VERSION_MAJOR}")
            set("${OUTPUT_VAR}" TRUE PARENT_SCOPE)
        else()
            set("${OUTPUT_VAR}" FALSE PARENT_SCOPE)
            message(WARNING "
            ** You are using different compiler idetifications for Fortran and C!
            *  This might lead to undefined behavior!")
        endif()
    endif()
endfunction()
