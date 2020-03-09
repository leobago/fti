add_library(jerasure OBJECT
    src/galois.c
    src/gf.c
    src/gf_cpu.c
    src/gf_w128.c
    src/gf_w16.c
    src/gf_w32.c
    src/gf_w4.c
    src/gf_w64.c
    src/gf_w8.c
    src/gf_wgen.c
    src/jerasure.c
)
target_include_directories(jerasure PRIVATE include)
set_property(TARGET jerasure PROPERTY POSITION_INDEPENDENT_CODE True)
