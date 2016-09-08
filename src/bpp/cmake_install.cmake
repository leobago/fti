# Install script for directory: /home/kellekai/Dokumente/BSC/WORK/FTI_REPOS/debug/vendor/bpp

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/kellekai/Dokumente/BSC/Training/FTI")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE PROGRAM FILES "/home/kellekai/Dokumente/BSC/WORK/FTI_REPOS/debug/src/bpp/bpp")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/Bpp/include/" TYPE DIRECTORY FILES "/home/kellekai/Dokumente/BSC/WORK/FTI_REPOS/debug/vendor/bpp/include/" FILES_MATCHING REGEX "/[^/]*\\.bpp\\.sh$")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/Bpp" TYPE FILE FILES "/home/kellekai/Dokumente/BSC/WORK/FTI_REPOS/debug/src/bpp/CMakeFiles/bpp.mk")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/CMake/Bpp" TYPE FILE FILES
    "/home/kellekai/Dokumente/BSC/WORK/FTI_REPOS/debug/vendor/bpp/cmake/Bpp.cmake"
    "/home/kellekai/Dokumente/BSC/WORK/FTI_REPOS/debug/vendor/bpp/cmake/TestFortType.cmake"
    "/home/kellekai/Dokumente/BSC/WORK/FTI_REPOS/debug/src/bpp/CMakeFiles/BppConfig.cmake"
    "/home/kellekai/Dokumente/BSC/WORK/FTI_REPOS/debug/src/bpp/BppConfigVersion.cmake"
    )
endif()

