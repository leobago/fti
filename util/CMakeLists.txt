set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

add_library(ftiUtil.static STATIC FTI_UtilList.c  FTI_UtilLowLevel.c FTI_UtilAPI.c
    		$<TARGET_OBJECTS:iniparser> $<TARGET_OBJECTS:jerasure>)
add_library(ftiUtil.shared SHARED FTI_UtilList.c  FTI_UtilLowLevel.c FTI_UtilAPI.c
    		$<TARGET_OBJECTS:iniparser> $<TARGET_OBJECTS:jerasure>)

#add_executable(FTIutil FTI_UtilList.c  FTI_UtilLowLevel.c FTI_UtilAPI.c $<TARGET_OBJECTS:iniparser> ${OPENSSL_LIBRARIES})
#target_link_libraries(FTIutil "${OPENSSL_LIBRARIES}")
set_target_properties(ftiUtil.static ftiUtil.shared PROPERTIES POSITION_INDEPENDENT_CODE True)
set_property(TARGET ftiUtil.static ftiUtil.shared 	PROPERTY OUTPUT_NAME ftiUtil)

set(FTI_TARGETS ftiUtil.static ftiUtil.shared)
install(TARGETS ftiUtil.static ftiUtil.shared DESTINATION lib)
install(FILES "FTI_UtilAPI.h" DESTINATION include)

set_source_files_properties(FTI_UtilAPI.c PROPERTIES CYTHON_IS_CXX FALSE)

add_library(hello.shared SHARED hello.c)

cython_add_module(HELLO hello.pyx  )
#cython_add_module(HELLO hello.pyx  $<TARGET_OBJECTS:iniparser> $<TARGET_OBJECTS:jerasure> )
target_link_libraries(HELLO "${OPENSSL_LIBRARIES}" hello.shared )
install(TARGETS HELLO DESTINATION lib)

