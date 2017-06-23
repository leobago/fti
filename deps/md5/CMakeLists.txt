set(SRC_md5
    md5.h md5.c)

append_property(SOURCE ${SRC_md5}
    PROPERTY COMPILE_FLAGS "-fPIC")

add_library(md5 OBJECT ${SRC_md5})
