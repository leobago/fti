add_executable(addInArray addInArray.c)
target_link_libraries(addInArray fti.static)

add_executable(diffSizes diffSizes.c)
target_link_libraries(diffSizes fti.static)

add_executable(heatdis heatdis.c)
target_link_libraries(heatdis fti.static)

add_executable(nodeFlag nodeFlag.c)
target_link_libraries(nodeFlag fti.static)

add_executable(corrupt corrupt.c)
target_link_libraries(corrupt fti.static)

add_subdirectory(local)

file(COPY configs DESTINATION .)
file(COPY patterns DESTINATION .)
file(COPY tests.sh DESTINATION .)
file(COPY heartbeat.sh DESTINATION .)
