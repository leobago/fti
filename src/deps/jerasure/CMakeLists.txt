file(GLOB JERASURE_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c")

add_library(jerasure OBJECT ${JERASURE_SRC})
target_include_directories(jerasure PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
set_property(TARGET jerasure PROPERTY POSITION_INDEPENDENT_CODE True)
