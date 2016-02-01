cmake_minimum_required(VERSION 2.8)

set(SRC_jerasure
    galois.h galois.c
    jerasure.h jerasure.c)

add_library(jerasure ${SRC_jerasure})
