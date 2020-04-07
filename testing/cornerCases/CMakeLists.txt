add_executable(consistency consistency.c)
target_link_libraries(consistency fti.static)

file(COPY corners.sh DESTINATION .)