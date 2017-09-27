file(GLOB JERASURE_SRC "${CMAKE_SOURCE_DIR}/deps/jerasure/src/*.c")
file(GLOB JERASURE_INC "${CMAKE_SOURCE_DIR}/deps/jerasure/include/*.h")

append_property(SOURCE ${JERASURE_SRC} PROPERTY COMPILE_FLAGS "-fPIC")

add_library(jerasure OBJECT ${JERASURE_SRC})
target_include_directories(jerasure PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
