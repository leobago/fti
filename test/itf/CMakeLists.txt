# ITF install directory
set(itf_dir ${CMAKE_CURRENT_BINARY_DIR})

# ITF template configuration file source and installed names
set(itf_config_template_src fti_template.cfg)
set(itf_config_template ${itf_dir}/${itf_config_template_src})

# All ITF files
set(itf_files engine api variables ${itf_config_template_src})

# Installation procedure
install(DIRECTORY DESTINATION ${itf_dir})

foreach(itf_file ${itf_files})
    configure_file(${itf_file} ${itf_file} @ONLY)
endforeach(itf_file)