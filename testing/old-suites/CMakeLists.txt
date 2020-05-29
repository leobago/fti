set(IO 1)

if (ENABLE_GPU)
  add_subdirectory(cuda)
endif()

add_subdirectory(local)
