set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

add_library(ftiUtil.static STATIC FTI_UtilList.c  FTI_UtilLowLevel.c FTI_UtilAPI.c
    		$<TARGET_OBJECTS:iniparser> $<TARGET_OBJECTS:jerasure>)
add_library(ftiUtil.shared SHARED FTI_UtilList.c  FTI_UtilLowLevel.c FTI_UtilAPI.c
    		$<TARGET_OBJECTS:iniparser> $<TARGET_OBJECTS:jerasure>)


add_executable(FTI_exportAllVars.exe FTI_exportAllVarsMPIO.c $<TARGET_OBJECTS:iniparser> $<TARGET_OBJECTS:jerasure>)

install(FILES "FTI_UtilAPI.h" DESTINATION include)


set_target_properties(ftiUtil.static ftiUtil.shared PROPERTIES POSITION_INDEPENDENT_CODE True)
set_property(TARGET ftiUtil.static ftiUtil.shared  PROPERTY OUTPUT_NAME ftiUtil)

set(FTI_TARGETS ftiUtil.static ftiUtil.shared  )
install(TARGETS ftiUtil.static ftiUtil.shared  DESTINATION lib)

target_link_libraries(FTI_exportAllVars.exe ftiUtil.shared ${MPI_C_LIBRARIES} m crypto)
set_property(TARGET FTI_exportAllVars.exe APPEND PROPERTY COMPILE_FLAGS ${MPI_C_COMPILE_FLAGS})
set_property(TARGET FTI_exportAllVars.exe APPEND PROPERTY LINK_FLAGS ${MPI_C_LINK_FLAGS} )
set_property(TARGET FTI_exportAllVars.exe APPEND PROPERTY LINK_FLAGS   )

install(TARGETS FTI_exportAllVars.exe DESTINATION bin) 



