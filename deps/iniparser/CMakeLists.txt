add_library(iniparser OBJECT 
    dictionary.c
    iniparser.c
)
set_property(TARGET iniparser PROPERTY POSITION_INDEPENDENT_CODE True)
