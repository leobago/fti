add_library(md5 OBJECT
    md5.c
)
set_property(TARGET md5 PROPERTY POSITION_INDEPENDENT_CODE True)
