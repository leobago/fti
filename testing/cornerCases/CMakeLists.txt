add_executable(consistency.exe consistency.c)
target_link_libraries(consistency.exe fti.static)

file(COPY corners.sh DESTINATION .)