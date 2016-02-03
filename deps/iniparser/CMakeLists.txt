set(SRC_iniparser
    dictionary.h dictionary.c
    iniparser.h iniparser.c)

append_property(SOURCE ${SRC_iniparser}
    PROPERTY COMPILE_FLAGS "-fPIC")

add_library(iniparser ${SRC_iniparser})
