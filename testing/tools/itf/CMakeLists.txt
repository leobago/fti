# --------------- Definition of ITF Installation-time variables ---------------

set(itf_dir ${testing_dir}/itf CACHE INTERNAL itf_dir)
set(itf_src_dir ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(itf_mod_dir ${itf_dir}/modules)
set(fti_cfg_template ${itf_dir}/resources/fti_template.cfg)

# FTI module configuration based on installation flags
set(fti_io_ids "1 2 3")
if(ENABLE_SIONLIB)
  set(fti_io_ids "${fti_io_ids} 4")
endif()
if (ENABLE_HDF5)
  set(fti_io_ids "${fti_io_ids} 5")
endif()

# ----------------------------- ITF Configuration -----------------------------

# Configure ITF modules
configure_file(modules/fti ${itf_mod_dir}/fti @ONLY)

# Install ITF resources
install(FILES resources/fti_template.cfg DESTINATION ${itf_dir}/resources)

# ------------------------------ ITF Testdriver -------------------------------

add_subdirectory(testdriver)
