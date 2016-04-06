set(SRC_jerasure
    galois.h galois.c
    jerasure.h jerasure.c)

append_property(SOURCE ${SRC_jerasure}
    PROPERTY COMPILE_FLAGS "-fPIC")

add_library(jerasure OBJECT ${SRC_jerasure})
