# --------------- Definition of ITF Installation-time variables ---------------

set(fti_io_ids "1 2 3")

if(ENABLE_SIONLIB)
  set(fti_io_ids "${fti_io_ids} 4")
endif()

if (ENABLE_HDF5)
  set(fti_io_ids "${fti_io_ids} 5")
endif()

set(itf_dir ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL itf_dir)
set(itf_src_dir ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(itf_mod_dir ${CMAKE_CURRENT_BINARY_DIR}/modules)

# ----------------------------- ITF source files ------------------------------

set(itf_modules
  fti
  parameters
)

set(itf_apps
  testdriver
)

# ----- ITF configuration ----

set(fti_cfg_template ${itf_dir}/resources/fti_template.cfg)
install(FILES resources/fti_template.cfg DESTINATION ${itf_dir}/resources)

# Configure the ITF modules
file(MAKE_DIRECTORY ${itf_mod_dir})
foreach(mod ${itf_modules})
  configure_file(modules/${mod} ${itf_mod_dir}/${mod} @ONLY)
endforeach(mod)

# Configure the ITF applications
foreach(app ${itf_apps})
  configure_file(${app} ${app} @ONLY)
endforeach(app)

set(itf_run_cmd ${itf_dir}/testdriver CACHE INTERNAL itf_run_cmd)
