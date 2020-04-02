set(itf_dir ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL itf_dir)

# --------------- Definition of ITF Installation-time variables ---------------

set(fti_io_names "POSIX MPIIO FTIFF")
set(fti_io_ids "1 2 3")

if(ENABLE_SIONLIB)
  set(fti_io_names "${fti_io_names} SIONLIB")
  set(fti_io_ids "${fti_io_ids} 4")
endif()

if (ENABLE_HDF5)
  set(fti_io_names "${fti_io_names} HDF5")
  set(fti_io_ids "${fti_io_ids} 5")
endif()

# -------- ITF template configuration file source and installed names ---------

set(itf_config_template_src fti_template.cfg)
set(itf_config_template ${itf_dir}/${itf_config_template_src})

# -------------------------------- Install ITF --------------------------------
set(itf_files engine api variables testdriver ${itf_config_template_src})

install(DIRECTORY DESTINATION ${itf_dir})
foreach(itf_file ${itf_files})
    configure_file(${itf_file} ${itf_file} @ONLY)
endforeach(itf_file)

set(itf_run_cmd ${itf_dir}/testdriver CACHE INTERNAL itf_run_cmd)