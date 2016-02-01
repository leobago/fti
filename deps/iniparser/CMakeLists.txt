cmake_minimum_required(VERSION 2.8)

set(SRC_iniparser
    dictionary.h dictionary.c
    iniparser.h iniparser.c)

add_library(iniparser ${SRC_iniparser})
