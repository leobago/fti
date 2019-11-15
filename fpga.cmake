# this one is important
SET(CMAKE_SYSTEM_NAME Linux)
#this one not so much
SET(CMAKE_SYSTEM_VERSION 1)


# specify the cross compiler
SET(CMAKE_C_COMPILER   arm-linux-gnueabihf-gcc)
SET(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)

# where is the target environment
SET(CMAKE_FIND_ROOT_PATH  /usr/arm-linux-gnueabihf/ /home/ompssATfpga/install-arm/)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE /usr/arm-linux-gnueabihf/include/ /home/ompssATfpga/install-arm/include/)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE /usr/arm-linux-gnueabihf/ /home/ompssATfpga/install-arm/)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM /usr/arm-linux-gnueabihf/ /home/ompssATfpga/install-arm/bin/)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY /usr/arm-linux-gnueabihf/lib/ /home/ompssATfpga/install-arm/lib/)

# search for programs in the build host directories
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

