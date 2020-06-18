set(SRC_iniparser
    dictionary.h dictionary.c
    iniparser.h iniparser.c)

set_property(SOURCE ${SRC_iniparser} APPEND PROPERTY COMPILE_FLAGS "-fPIC")

add_library(iniparser OBJECT ${SRC_iniparser})
