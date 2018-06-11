add_executable(ckptHierarchy ckptHierarchy.c)
target_link_libraries(ckptHierarchy fti.static)

file(COPY ckptHierarchy.sh DESTINATION .)