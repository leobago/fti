set(itf_run_cmd "${itf_dir}/testdriver" CACHE INTERNAL itf_run_cmd)

configure_file(testdriver_build ${itf_dir}/testdriver @ONLY)
