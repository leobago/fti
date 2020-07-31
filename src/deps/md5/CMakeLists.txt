set(SRC_md5 md5.h md5.c)
set_property(SOURCE ${SRC_md5} APPEND PROPERTY COMPILE_FLAGS "-fPIC")
add_library(md5 OBJECT ${SRC_md5})
