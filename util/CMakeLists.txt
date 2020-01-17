set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
add_executable(FTIutil tool_api.c $<TARGET_OBJECTS:iniparser> )
